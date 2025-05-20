# Latest Large Language Model Attack Papers
**update at 2025-05-20 10:31:01**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks**

调查LLM作为法官架构对预算注入攻击的脆弱性 cs.CL

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.13348v1) [paper-pdf](http://arxiv.org/pdf/2505.13348v1)

**Authors**: Narek Maloyan, Bislan Ashinov, Dmitry Namiot

**Abstract**: Large Language Models (LLMs) are increasingly employed as evaluators (LLM-as-a-Judge) for assessing the quality of machine-generated text. This paradigm offers scalability and cost-effectiveness compared to human annotation. However, the reliability and security of such systems, particularly their robustness against adversarial manipulations, remain critical concerns. This paper investigates the vulnerability of LLM-as-a-Judge architectures to prompt-injection attacks, where malicious inputs are designed to compromise the judge's decision-making process. We formalize two primary attack strategies: Comparative Undermining Attack (CUA), which directly targets the final decision output, and Justification Manipulation Attack (JMA), which aims to alter the model's generated reasoning. Using the Greedy Coordinate Gradient (GCG) optimization method, we craft adversarial suffixes appended to one of the responses being compared. Experiments conducted on the MT-Bench Human Judgments dataset with open-source instruction-tuned LLMs (Qwen2.5-3B-Instruct and Falcon3-3B-Instruct) demonstrate significant susceptibility. The CUA achieves an Attack Success Rate (ASR) exceeding 30\%, while JMA also shows notable effectiveness. These findings highlight substantial vulnerabilities in current LLM-as-a-Judge systems, underscoring the need for robust defense mechanisms and further research into adversarial evaluation and trustworthiness in LLM-based assessment frameworks.

摘要: 大型语言模型（LLM）越来越多地被用作评估器（LLM as-a-Judge）来评估机器生成文本的质量。与人类注释相比，该范式提供了可扩展性和成本效益。然而，此类系统的可靠性和安全性，特别是它们对对抗性操纵的鲁棒性，仍然是关键问题。本文研究了LLM as-a-Judge架构对预算注入攻击的脆弱性，其中恶意输入旨在损害法官的决策过程。我们正式化了两种主要的攻击策略：比较挖掘攻击（CUA），直接针对最终决策输出，和合理化操纵攻击（JMA），旨在改变模型生成的推理。使用贪婪坐标梯度（GCG）优化方法，我们制作附加到正在比较的一个响应上的对抗后缀。在MT-Bench Human Judgments数据集上使用开源描述调整的LLM（Qwen 2.5 - 3B-Direct和Falcon 3 - 3B-Direct）进行的实验证明了显着的易感性。CUA的攻击成功率（ASB）超过30%，而JMA也表现出显着的有效性。这些发现凸显了当前法学硕士作为法官系统中的重大漏洞，强调了强大的防御机制以及对基于法学硕士的评估框架中的对抗性评估和可信度进行进一步研究的必要性。



## **2. Concept-Level Explainability for Auditing & Steering LLM Responses**

审计和指导LLM响应的概念级解释性 cs.CL

9 pages, 7 figures, Submission to Neurips 2025

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.07610v2) [paper-pdf](http://arxiv.org/pdf/2505.07610v2)

**Authors**: Kenza Amara, Rita Sevastjanova, Mennatallah El-Assady

**Abstract**: As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior.

摘要: 随着大型语言模型（LLM）的广泛部署，对其安全性和一致性的担忧日益加剧。引导LLM行为（例如减轻偏见或防范越狱）的一种方法是识别提示的哪些部分影响模型输出的特定方面。令牌级归因方法提供了一个有希望的解决方案，但在文本生成方面仍然很困难，分别解释输出中每个令牌的存在，而不是整个LLM响应的底层语义。我们引入ConceptX，这是一种模型不可知的概念级解释方法，可以识别概念，即提示中语义丰富的标记，并根据输出的语义相似性为其分配重要性。与当前的代币级方法不同，ConceptX还提供通过就地代币替换来保持上下文完整性，并支持灵活的解释目标，例如性别偏见。ConceptX通过发现偏见的来源来实现审计，并通过修改提示以改变情绪或减少LLM响应的危害性来实现引导，而无需再培训。在三个LLM中，ConceptX在忠诚度和人性化方面都优于TokenSHAP等代币级方法。随机编辑的引导任务使情绪转变提高了0.252和0.131，攻击成功率从0.463降低到0.242，优于归因和重述基线。虽然及时的工程和自我解释方法有时会产生更安全的响应，但ConceptX为提高LLM安全性和一致性提供了一种透明且忠实的替代方案，展示了基于属性的解释在指导LLM行为方面的实际价值。



## **3. The Hidden Dangers of Browsing AI Agents**

浏览人工智能代理的隐藏危险 cs.CR

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.13076v1) [paper-pdf](http://arxiv.org/pdf/2505.13076v1)

**Authors**: Mykyta Mudryi, Markiyan Chaklosh, Grzegorz Wójcik

**Abstract**: Autonomous browsing agents powered by large language models (LLMs) are increasingly used to automate web-based tasks. However, their reliance on dynamic content, tool execution, and user-provided data exposes them to a broad attack surface. This paper presents a comprehensive security evaluation of such agents, focusing on systemic vulnerabilities across multiple architectural layers. Our work outlines the first end-to-end threat model for browsing agents and provides actionable guidance for securing their deployment in real-world environments. To address discovered threats, we propose a defense in depth strategy incorporating input sanitization, planner executor isolation, formal analyzers, and session safeguards. These measures protect against both initial access and post exploitation attack vectors. Through a white box analysis of a popular open source project, Browser Use, we demonstrate how untrusted web content can hijack agent behavior and lead to critical security breaches. Our findings include prompt injection, domain validation bypass, and credential exfiltration, evidenced by a disclosed CVE and a working proof of concept exploit.

摘要: 由大型语言模型（LLM）支持的自主浏览代理越来越多地用于自动化基于Web的任务。然而，它们对动态内容、工具执行和用户提供的数据的依赖使它们面临广泛的攻击面。本文对此类代理进行了全面的安全评估，重点关注跨多个体系结构层的系统漏洞。我们的工作概述了浏览代理的第一个端到端威胁模型，并为确保其在现实世界环境中的部署提供了可操作的指导。为了解决发现的威胁，我们提出了一种深度防御策略，其中包括输入清理、计划执行者隔离、正式分析器和会话保护措施。这些措施可以防止初始访问和利用后攻击媒介。通过对流行的开源项目“浏览器使用”的白盒分析，我们展示了不受信任的Web内容如何劫持代理行为并导致严重的安全漏洞。我们的发现包括即时注入、域验证绕过和凭证外流，并通过公开的UTE和概念利用的有效证明来证明。



## **4. Evaluatiing the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset**

评估LLM安全解决方案的有效性：Palit基准数据集 cs.CR

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.13028v1) [paper-pdf](http://arxiv.org/pdf/2505.13028v1)

**Authors**: Sayon Palit, Daniel Woods

**Abstract**: Large Language Models (LLMs) are increasingly integrated into critical systems in industries like healthcare and finance. Users can often submit queries to LLM-enabled chatbots, some of which can enrich responses with information retrieved from internal databases storing sensitive data. This gives rise to a range of attacks in which a user submits a malicious query and the LLM-system outputs a response that creates harm to the owner, such as leaking internal data or creating legal liability by harming a third-party. While security tools are being developed to counter these threats, there is little formal evaluation of their effectiveness and usability. This study addresses this gap by conducting a thorough comparative analysis of LLM security tools. We identified 13 solutions (9 closed-source, 4 open-source), but only 7 were evaluated due to a lack of participation by proprietary model owners.To evaluate, we built a benchmark dataset of malicious prompts, and evaluate these tools performance against a baseline LLM model (ChatGPT-3.5-Turbo). Our results show that the baseline model has too many false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard emerged as the best overall tools showcasing the tradeoff between usability and performance. The study concluded with recommendations for greater transparency among closed source providers, improved context-aware detections, enhanced open-source engagement, increased user awareness, and the adoption of more representative performance metrics.

摘要: 大型语言模型（LLM）越来越多地集成到医疗保健和金融等行业的关键系统中。用户通常可以向支持LLM的聊天机器人提交查询，其中一些可以使用从存储敏感数据的内部数据库检索的信息来丰富响应。这会引发一系列攻击，其中用户提交恶意查询，LLM系统输出对所有者造成伤害的响应，例如泄露内部数据或通过伤害第三方而产生法律责任。虽然正在开发安全工具来应对这些威胁，但对其有效性和可用性的正式评估很少。本研究通过对LLM安全工具进行彻底的比较分析来解决这一差距。我们确定了13个解决方案（9个封闭源，4个开放源），但由于缺乏专有模型所有者的参与，只评估了7个。为了评估，我们构建了恶意提示的基准数据集，并根据基线LLM模型（ChatGPT-3.5-Turbo）评估这些工具的性能。我们的结果表明，基线模型存在太多假阳性，无法用于此任务。Lakera Guard和ProtectAI LLM Guard成为展示可用性和性能之间权衡的最佳整体工具。该研究最后提出了提高闭源提供商透明度、改进上下文感知检测、增强开源参与度、提高用户意识以及采用更具代表性的性能指标的建议。



## **5. From Assistants to Adversaries: Exploring the Security Risks of Mobile LLM Agents**

从助理到对手：探索移动LLM代理的安全风险 cs.CR

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12981v1) [paper-pdf](http://arxiv.org/pdf/2505.12981v1)

**Authors**: Liangxuan Wu, Chao Wang, Tianming Liu, Yanjie Zhao, Haoyu Wang

**Abstract**: The growing adoption of large language models (LLMs) has led to a new paradigm in mobile computing--LLM-powered mobile AI agents--capable of decomposing and automating complex tasks directly on smartphones. However, the security implications of these agents remain largely unexplored. In this paper, we present the first comprehensive security analysis of mobile LLM agents, encompassing three representative categories: System-level AI Agents developed by original equipment manufacturers (e.g., YOYO Assistant), Third-party Universal Agents (e.g., Zhipu AI AutoGLM), and Emerging Agent Frameworks (e.g., Alibaba Mobile Agent). We begin by analyzing the general workflow of mobile agents and identifying security threats across three core capability dimensions: language-based reasoning, GUI-based interaction, and system-level execution. Our analysis reveals 11 distinct attack surfaces, all rooted in the unique capabilities and interaction patterns of mobile LLM agents, and spanning their entire operational lifecycle. To investigate these threats in practice, we introduce AgentScan, a semi-automated security analysis framework that systematically evaluates mobile LLM agents across all 11 attack scenarios. Applying AgentScan to nine widely deployed agents, we uncover a concerning trend: every agent is vulnerable to targeted attacks. In the most severe cases, agents exhibit vulnerabilities across eight distinct attack vectors. These attacks can cause behavioral deviations, privacy leakage, or even full execution hijacking. Based on these findings, we propose a set of defensive design principles and practical recommendations for building secure mobile LLM agents. Our disclosures have received positive feedback from two major device vendors. Overall, this work highlights the urgent need for standardized security practices in the fast-evolving landscape of LLM-driven mobile automation.

摘要: 大型语言模型（LLM）的日益采用催生了移动计算领域的一种新范式--LLM支持的移动人工智能代理--能够直接在智能手机上分解和自动化复杂任务。然而，这些特工的安全影响在很大程度上仍未得到探讨。在本文中，我们首次对移动LLM代理进行了全面的安全分析，涵盖三个代表性类别：由原始设备制造商开发的系统级AI代理（例如，YOYO助理）、第三方环球代理（例如，知普AI AutoGLM）和新兴代理框架（例如，阿里巴巴移动代理）。我们首先分析移动代理的一般工作流程，并识别三个核心能力维度的安全威胁：基于语言的推理、基于图形用户界面的交互和系统级执行。我们的分析揭示了11种不同的攻击表面，所有这些都植根于移动LLM代理的独特功能和交互模式，并跨越其整个运营生命周期。为了在实践中调查这些威胁，我们引入了AgentScan，这是一个半自动安全分析框架，可以系统地评估所有11种攻击场景中的移动LLM代理。将AgentScan应用于九个广泛部署的代理，我们发现了一个令人担忧的趋势：每个代理都容易受到有针对性的攻击。在最严重的情况下，代理在八个不同的攻击载体上表现出漏洞。这些攻击可能会导致行为偏差、隐私泄露，甚至完全执行劫持。基于这些发现，我们提出了一套用于构建安全移动LLM代理的防御设计原则和实用建议。我们的披露得到了两家主要设备供应商的积极反馈。总体而言，这项工作凸显了在LLM驱动的移动自动化快速发展的环境中对标准化安全实践的迫切需要。



## **6. "Yes, My LoRD." Guiding Language Model Extraction with Locality Reinforced Distillation**

“是的，我的爱人。“利用局部强化蒸馏提取引导语言模型 cs.CR

To appear at ACL 25 main conference

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2409.02718v3) [paper-pdf](http://arxiv.org/pdf/2409.02718v3)

**Authors**: Zi Liang, Qingqing Ye, Yanyun Wang, Sen Zhang, Yaxin Xiao, Ronghua Li, Jianliang Xu, Haibo Hu

**Abstract**: Model extraction attacks (MEAs) on large language models (LLMs) have received increasing attention in recent research. However, existing attack methods typically adapt the extraction strategies originally developed for deep neural networks (DNNs). They neglect the underlying inconsistency between the training tasks of MEA and LLM alignment, leading to suboptimal attack performance. To tackle this issue, we propose Locality Reinforced Distillation (LoRD), a novel model extraction algorithm specifically designed for LLMs. In particular, LoRD employs a newly defined policy-gradient-style training task that utilizes the responses of victim model as the signal to guide the crafting of preference for the local model. Theoretical analyses demonstrate that I) The convergence procedure of LoRD in model extraction is consistent with the alignment procedure of LLMs, and II) LoRD can reduce query complexity while mitigating watermark protection through our exploration-based stealing. Extensive experiments validate the superiority of our method in extracting various state-of-the-art commercial LLMs. Our code is available at: https://github.com/liangzid/LoRD-MEA .

摘要: 在最近的研究中，对大型语言模型（LLM）的模型提取攻击（MEAs）受到越来越多的关注。然而，现有的攻击方法通常会适应最初为深度神经网络（DNN）开发的提取策略。他们忽视了EMA和LLM对齐训练任务之间的潜在不一致性，导致攻击性能次优。为了解决这个问题，我们提出了局部强化蒸馏（LoRD），这是一种专门为LLM设计的新型模型提取算法。特别是，LoRD采用了新定义的政策梯度式培训任务，该任务利用受害者模型的反应作为信号来指导制定对本地模型的偏好。理论分析表明，I）LoRD在模型提取中的收敛过程与LLM的对齐过程一致，II）LoRD可以降低查询复杂性，同时通过我们基于探索的窃取来减轻水印保护。大量实验验证了我们的方法在提取各种最先进的商业LLM方面的优越性。我们的代码可访问：https://github.com/liangzid/LoRD-MEA。



## **7. Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks?**

低等级适应是否会导致针对训练时间攻击的鲁棒性较低？ cs.LG

To appear at ICML 25

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12871v1) [paper-pdf](http://arxiv.org/pdf/2505.12871v1)

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Ronghua Li

**Abstract**: Low rank adaptation (LoRA) has emerged as a prominent technique for fine-tuning large language models (LLMs) thanks to its superb efficiency gains over previous methods. While extensive studies have examined the performance and structural properties of LoRA, its behavior upon training-time attacks remain underexplored, posing significant security risks. In this paper, we theoretically investigate the security implications of LoRA's low-rank structure during fine-tuning, in the context of its robustness against data poisoning and backdoor attacks. We propose an analytical framework that models LoRA's training dynamics, employs the neural tangent kernel to simplify the analysis of the training process, and applies information theory to establish connections between LoRA's low rank structure and its vulnerability against training-time attacks. Our analysis indicates that LoRA exhibits better robustness to backdoor attacks than full fine-tuning, while becomes more vulnerable to untargeted data poisoning due to its over-simplified information geometry. Extensive experimental evaluations have corroborated our theoretical findings.

摘要: 低秩自适应（LoRA）已经成为一种用于微调大型语言模型（LLM）的突出技术，这要归功于它比以前的方法具有更高的效率。虽然广泛的研究已经检查了LoRA的性能和结构特性，但其在训练时间攻击时的行为仍然没有得到充分的研究，从而带来了重大的安全风险。在本文中，我们从理论上研究了LoRA的低秩结构在微调过程中的安全性影响，在其对数据中毒和后门攻击的鲁棒性的背景下。我们提出了一个分析框架，模型LoRA的训练动态，采用神经正切内核来简化训练过程的分析，并应用信息论建立LoRA的低秩结构和它对训练时间攻击的脆弱性之间的连接。我们的分析表明，LoRA对后门攻击表现出比完全微调更好的鲁棒性，同时由于其过于简化的信息几何结构，更容易受到非目标数据中毒的影响。广泛的实验评估证实了我们的理论发现。



## **8. LLMPot: Dynamically Configured LLM-based Honeypot for Industrial Protocol and Physical Process Emulation**

LLMPot：动态配置的基于LLM的蜜罐，用于工业协议和物理流程仿真 cs.CR

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2405.05999v3) [paper-pdf](http://arxiv.org/pdf/2405.05999v3)

**Authors**: Christoforos Vasilatos, Dunia J. Mahboobeh, Hithem Lamri, Manaar Alam, Michail Maniatakos

**Abstract**: Industrial Control Systems (ICS) are extensively used in critical infrastructures ensuring efficient, reliable, and continuous operations. However, their increasing connectivity and addition of advanced features make them vulnerable to cyber threats, potentially leading to severe disruptions in essential services. In this context, honeypots play a vital role by acting as decoy targets within ICS networks, or on the Internet, helping to detect, log, analyze, and develop mitigations for ICS-specific cyber threats. Deploying ICS honeypots, however, is challenging due to the necessity of accurately replicating industrial protocols and device characteristics, a crucial requirement for effectively mimicking the unique operational behavior of different industrial systems. Moreover, this challenge is compounded by the significant manual effort required in also mimicking the control logic the PLC would execute, in order to capture attacker traffic aiming to disrupt critical infrastructure operations. In this paper, we propose LLMPot, a novel approach for designing honeypots in ICS networks harnessing the potency of Large Language Models (LLMs). LLMPot aims to automate and optimize the creation of realistic honeypots with vendor-agnostic configurations, and for any control logic, aiming to eliminate the manual effort and specialized knowledge traditionally required in this domain. We conducted extensive experiments focusing on a wide array of parameters, demonstrating that our LLM-based approach can effectively create honeypot devices implementing different industrial protocols and diverse control logic.

摘要: 工业控制系统（ICS）广泛用于关键基础设施，确保高效、可靠和连续的运营。然而，它们不断增加的连接性和添加的高级功能使它们容易受到网络威胁的影响，可能导致基本服务的严重中断。在这种情况下，蜜罐发挥着至关重要的作用，充当ICS网络内或互联网上的诱饵目标，帮助检测、记录、分析和开发针对ICS特定网络威胁的缓解措施。然而，部署ICS蜜罐具有挑战性，因为需要准确地复制工业协议和设备特征，这是有效模仿不同工业系统独特操作行为的关键要求。此外，模仿PLC将执行的控制逻辑以捕获旨在破坏关键基础设施运营的攻击者流量所需的大量手动工作使这一挑战变得更加复杂。在本文中，我们提出了LLMPot，这是一种在ICS网络中设计蜜罐的新颖方法，利用大型语言模型（LLM）的能力。LLMPot旨在自动化和优化具有供应商不可知配置的现实蜜罐的创建，并适用于任何控制逻辑，旨在消除该领域传统上所需的手动工作和专业知识。我们针对广泛的参数进行了广泛的实验，证明我们基于LLM的方法可以有效地创建实施不同工业协议和不同控制逻辑的蜜罐设备。



## **9. Forewarned is Forearmed: A Survey on Large Language Model-based Agents in Autonomous Cyberattacks**

预先警告就是预先武装：自主网络攻击中基于大型语言模型的代理的调查 cs.NI

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12786v1) [paper-pdf](http://arxiv.org/pdf/2505.12786v1)

**Authors**: Minrui Xu, Jiani Fan, Xinyu Huang, Conghao Zhou, Jiawen Kang, Dusit Niyato, Shiwen Mao, Zhu Han, Xuemin, Shen, Kwok-Yan Lam

**Abstract**: With the continuous evolution of Large Language Models (LLMs), LLM-based agents have advanced beyond passive chatbots to become autonomous cyber entities capable of performing complex tasks, including web browsing, malicious code and deceptive content generation, and decision-making. By significantly reducing the time, expertise, and resources, AI-assisted cyberattacks orchestrated by LLM-based agents have led to a phenomenon termed Cyber Threat Inflation, characterized by a significant reduction in attack costs and a tremendous increase in attack scale. To provide actionable defensive insights, in this survey, we focus on the potential cyber threats posed by LLM-based agents across diverse network systems. Firstly, we present the capabilities of LLM-based cyberattack agents, which include executing autonomous attack strategies, comprising scouting, memory, reasoning, and action, and facilitating collaborative operations with other agents or human operators. Building on these capabilities, we examine common cyberattacks initiated by LLM-based agents and compare their effectiveness across different types of networks, including static, mobile, and infrastructure-free paradigms. Moreover, we analyze threat bottlenecks of LLM-based agents across different network infrastructures and review their defense methods. Due to operational imbalances, existing defense methods are inadequate against autonomous cyberattacks. Finally, we outline future research directions and potential defensive strategies for legacy network systems.

摘要: 随着大型语言模型（LLM）的不断发展，基于LLM的代理已经超越被动聊天机器人，成为能够执行复杂任务的自治网络实体，包括网络浏览、恶意代码和欺骗性内容生成以及决策。通过显着减少时间、专业知识和资源，由LLM代理策划的人工智能辅助网络攻击导致了一种称为网络威胁通货膨胀的现象，其特征是攻击成本显着降低和攻击规模显着增加。为了提供可操作的防御见解，在本调查中，我们重点关注基于LLM的代理在不同网络系统中构成的潜在网络威胁。首先，我们介绍了基于LLM的网络攻击代理的能力，其中包括执行自主攻击策略，包括侦察、记忆、推理和行动，以及促进与其他代理或人类操作员的协作操作。基于这些功能，我们研究了基于LLM的代理发起的常见网络攻击，并比较了它们在不同类型网络（包括静态，移动和无基础设施模式）中的有效性。此外，我们分析了基于LLM的代理在不同的网络基础设施的威胁瓶颈，并审查其防御方法。由于操作不平衡，现有的防御方法不足以应对自主网络攻击。最后，我们概述了未来的研究方向和潜在的防御策略的遗留网络系统。



## **10. Language Models That Walk the Talk: A Framework for Formal Fairness Certificates**

直言不讳的语言模型：正式公平证书的框架 cs.AI

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12767v1) [paper-pdf](http://arxiv.org/pdf/2505.12767v1)

**Authors**: Danqing Chen, Tobias Ladner, Ahmed Rayen Mhadhbi, Matthias Althoff

**Abstract**: As large language models become integral to high-stakes applications, ensuring their robustness and fairness is critical. Despite their success, large language models remain vulnerable to adversarial attacks, where small perturbations, such as synonym substitutions, can alter model predictions, posing risks in fairness-critical areas, such as gender bias mitigation, and safety-critical areas, such as toxicity detection. While formal verification has been explored for neural networks, its application to large language models remains limited. This work presents a holistic verification framework to certify the robustness of transformer-based language models, with a focus on ensuring gender fairness and consistent outputs across different gender-related terms. Furthermore, we extend this methodology to toxicity detection, offering formal guarantees that adversarially manipulated toxic inputs are consistently detected and appropriately censored, thereby ensuring the reliability of moderation systems. By formalizing robustness within the embedding space, this work strengthens the reliability of language models in ethical AI deployment and content moderation.

摘要: 随着大型语言模型成为高风险应用程序的组成部分，确保其稳健性和公平性至关重要。尽管取得了成功，大型语言模型仍然容易受到对抗攻击，其中同义词替换等小扰动可能会改变模型预测，从而在性别偏见缓解等公平关键领域和安全关键领域带来风险，例如毒性检测。虽然已经探索了神经网络的形式验证，但其在大型语言模型中的应用仍然有限。这项工作提出了一个整体验证框架，以验证基于转换器的语言模型的稳健性，重点是确保性别公平性和不同性别相关术语的一致输出。此外，我们将这种方法扩展到毒性检测，提供正式保证，以一致地检测和适当审查敌对操纵的有毒输入，从而确保审核系统的可靠性。通过形式化嵌入空间内的鲁棒性，这项工作增强了语言模型在道德人工智能部署和内容审核中的可靠性。



## **11. BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models**

BackdoorLLM：大型语言模型后门攻击和防御的综合基准 cs.AI

22 pages

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2408.12798v2) [paper-pdf](http://arxiv.org/pdf/2408.12798v2)

**Authors**: Yige Li, Hanxun Huang, Yunhan Zhao, Xingjun Ma, Jun Sun

**Abstract**: Generative large language models (LLMs) have achieved state-of-the-art results on a wide range of tasks, yet they remain susceptible to backdoor attacks: carefully crafted triggers in the input can manipulate the model to produce adversary-specified outputs. While prior research has predominantly focused on backdoor risks in vision and classification settings, the vulnerability of LLMs in open-ended text generation remains underexplored. To fill this gap, we introduce BackdoorLLM (Our BackdoorLLM benchmark was awarded First Prize in the SafetyBench competition, https://www.mlsafety.org/safebench/winners, organized by the Center for AI Safety, https://safe.ai/.), the first comprehensive benchmark for systematically evaluating backdoor threats in text-generation LLMs. BackdoorLLM provides: (i) a unified repository of benchmarks with a standardized training and evaluation pipeline; (ii) a diverse suite of attack modalities, including data poisoning, weight poisoning, hidden-state manipulation, and chain-of-thought hijacking; (iii) over 200 experiments spanning 8 distinct attack strategies, 7 real-world scenarios, and 6 model architectures; (iv) key insights into the factors that govern backdoor effectiveness and failure modes in LLMs; and (v) a defense toolkit encompassing 7 representative mitigation techniques. Our code and datasets are available at https://github.com/bboylyg/BackdoorLLM. We will continuously incorporate emerging attack and defense methodologies to support the research in advancing the safety and reliability of LLMs.

摘要: 生成式大型语言模型（LLM）在广泛的任务上取得了最先进的结果，但它们仍然容易受到后门攻击：输入中精心制作的触发器可以操纵模型以产生对手指定的输出。虽然先前的研究主要集中在视觉和分类设置中的后门风险，但LLM在开放式文本生成中的脆弱性仍然没有得到充分的研究。为了填补这一空白，我们引入了BackdoorLLM（我们的BackdoorLLM基准测试在SafetyBench竞赛中获得一等奖，https：//www.mlsafety.org/safebench/winners，由AI安全中心组织，https：//safe.ai/.），系统评估文本生成LLM中后门威胁的第一个全面基准。BackdoorLLM提供：（i）一个统一的基准库，具有标准化的培训和评估管道;（ii）一套多样化的攻击模式，包括数据中毒，权重中毒，隐藏状态操纵和思想链劫持;（iii）超过200个实验，涵盖8种不同的攻击策略，7种真实场景和6种模型架构;（iv）对制约LLM后门有效性和故障模式的因素的关键见解;以及（v）包含7种代表性缓解技术的防御工具包。我们的代码和数据集可在https://github.com/bboylyg/BackdoorLLM上获取。我们将不断结合新兴的攻击和防御方法，以支持研究，提高LLM的安全性和可靠性。



## **12. Bullying the Machine: How Personas Increase LLM Vulnerability**

欺凌机器：角色扮演如何增加LLM漏洞 cs.AI

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12692v1) [paper-pdf](http://arxiv.org/pdf/2505.12692v1)

**Authors**: Ziwei Xu, Udit Sanghi, Mohan Kankanhalli

**Abstract**: Large Language Models (LLMs) are increasingly deployed in interactions where they are prompted to adopt personas. This paper investigates whether such persona conditioning affects model safety under bullying, an adversarial manipulation that applies psychological pressures in order to force the victim to comply to the attacker. We introduce a simulation framework in which an attacker LLM engages a victim LLM using psychologically grounded bullying tactics, while the victim adopts personas aligned with the Big Five personality traits. Experiments using multiple open-source LLMs and a wide range of adversarial goals reveal that certain persona configurations -- such as weakened agreeableness or conscientiousness -- significantly increase victim's susceptibility to unsafe outputs. Bullying tactics involving emotional or sarcastic manipulation, such as gaslighting and ridicule, are particularly effective. These findings suggest that persona-driven interaction introduces a novel vector for safety risks in LLMs and highlight the need for persona-aware safety evaluation and alignment strategies.

摘要: 大型语言模型（LLM）越来越多地部署在交互中，它们会被提示采用角色。本文研究了这种角色条件反射是否会影响欺凌下的模型安全性，欺凌是一种对抗性操纵，施加心理压力以迫使受害者服从攻击者。我们引入了一个模拟框架，其中攻击者LLM使用基于心理的欺凌策略与受害者LLM互动，而受害者则采用与五大人格特征一致的角色。使用多个开源LLM和广泛的对抗目标的实验表明，某些角色配置（例如减弱的宜人性或危险性）会显着增加受害者对不安全输出的易感性。涉及情感或讽刺操纵的欺凌策略，例如煤气灯和嘲笑，尤其有效。这些发现表明，个性驱动的交互为LLM中的安全风险引入了一种新的载体，并强调了对个性意识的安全评估和协调策略的必要性。



## **13. Use as Many Surrogates as You Want: Selective Ensemble Attack to Unleash Transferability without Sacrificing Resource Efficiency**

使用尽可能多的代理人：选择性发起攻击以释放可转让性，而不牺牲资源效率 cs.CV

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12644v1) [paper-pdf](http://arxiv.org/pdf/2505.12644v1)

**Authors**: Bo Yang, Hengwei Zhang, Jindong Wang, Yuchen Ren, Chenhao Lin, Chao Shen, Zhengyu Zhao

**Abstract**: In surrogate ensemble attacks, using more surrogate models yields higher transferability but lower resource efficiency. This practical trade-off between transferability and efficiency has largely limited existing attacks despite many pre-trained models are easily accessible online. In this paper, we argue that such a trade-off is caused by an unnecessary common assumption, i.e., all models should be identical across iterations. By lifting this assumption, we can use as many surrogates as we want to unleash transferability without sacrificing efficiency. Concretely, we propose Selective Ensemble Attack (SEA), which dynamically selects diverse models (from easily accessible pre-trained models) across iterations based on our new interpretation of decoupling within-iteration and cross-iteration model diversity.In this way, the number of within-iteration models is fixed for maintaining efficiency, while only cross-iteration model diversity is increased for higher transferability. Experiments on ImageNet demonstrate the superiority of SEA in various scenarios. For example, when dynamically selecting 4 from 20 accessible models, SEA yields 8.5% higher transferability than existing attacks under the same efficiency. The superiority of SEA also generalizes to real-world systems, such as commercial vision APIs and large vision-language models. Overall, SEA opens up the possibility of adaptively balancing transferability and efficiency according to specific resource requirements.

摘要: 在代理集成攻击中，使用更多代理模型会产生更高的可移植性，但资源效率较低。尽管许多预先训练的模型可以轻松在线访问，但可移植性和效率之间的这种实际权衡在很大程度上限制了现有的攻击。在本文中，我们认为这种权衡是由不必要的共同假设引起的，即，所有模型在迭代中都应该相同。通过取消这一假设，我们可以使用尽可能多的代理人，以释放可转移性，而不牺牲效率。具体来说，我们提出了选择性集合攻击（SEA），它基于我们对迭代内和跨迭代模型多样性脱钩的新解释，在迭代之间动态选择不同的模型（从易于访问的预训练模型中）。这样，迭代内模型的数量是固定的，以维持效率，而只有增加跨迭代模型多样性以获得更高的可移植性。ImageNet上的实验证明了SEA在各种场景下的优越性。例如，当从20个可访问模型中动态选择4个时，在相同效率下，SEA的可转移性比现有攻击高出8.5%。SEA的优势还推广到现实世界的系统，例如商业视觉API和大型视觉语言模型。总体而言，SEA开辟了根据特定资源要求自适应地平衡可转移性和效率的可能性。



## **14. PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation**

PoisonArena：揭露检索增强一代中的竞争中毒攻击 cs.IR

29 pages

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12574v1) [paper-pdf](http://arxiv.org/pdf/2505.12574v1)

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: https://github.com/yxf203/PoisonArena.

摘要: 检索增强生成（RAG）系统，广泛用于改善大型语言模型（LLM）的事实基础，越来越容易受到中毒攻击，其中对手将操纵的内容注入检索器的语料库。虽然以前的研究主要集中在单个攻击者的设置，但现实世界的场景往往涉及多个相互竞争的攻击者，这些攻击者的目标相互冲突。在这项工作中，我们介绍PoisonArena，第一个基准系统地研究和评估竞争中毒攻击在RAG。我们形式化的多攻击者威胁模型，攻击者争夺控制答案相同的查询使用互斥的错误信息。PoisonArena利用Bradley-Terry模型来量化每种方法在此类对抗环境中的竞争有效性。通过对Natural Questions和MS MARCO数据集的广泛实验，我们证明了许多孤立成功的攻击策略在竞争压力下失败。我们的研究结果强调了攻击成功率（SVR）和F1评分等传统评估指标的局限性，并强调了竞争性评估来评估现实世界攻击稳健性的必要性。PoisonArena提供了一个标准化的框架，可以在更现实的多对手条件下基准和开发未来的攻击和防御策略。项目页面：https://github.com/yxf203/PoisonArena。



## **15. A Survey of Attacks on Large Language Models**

大型语言模型攻击调查 cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12567v1) [paper-pdf](http://arxiv.org/pdf/2505.12567v1)

**Authors**: Wenrui Xu, Keshab K. Parhi

**Abstract**: Large language models (LLMs) and LLM-based agents have been widely deployed in a wide range of applications in the real world, including healthcare diagnostics, financial analysis, customer support, robotics, and autonomous driving, expanding their powerful capability of understanding, reasoning, and generating natural languages. However, the wide deployment of LLM-based applications exposes critical security and reliability risks, such as the potential for malicious misuse, privacy leakage, and service disruption that weaken user trust and undermine societal safety. This paper provides a systematic overview of the details of adversarial attacks targeting both LLMs and LLM-based agents. These attacks are organized into three phases in LLMs: Training-Phase Attacks, Inference-Phase Attacks, and Availability & Integrity Attacks. For each phase, we analyze the details of representative and recently introduced attack methods along with their corresponding defenses. We hope our survey will provide a good tutorial and a comprehensive understanding of LLM security, especially for attacks on LLMs. We desire to raise attention to the risks inherent in widely deployed LLM-based applications and highlight the urgent need for robust mitigation strategies for evolving threats.

摘要: 大型语言模型（LLM）和基于LLM的代理已被广泛部署在现实世界的广泛应用中，包括医疗诊断，财务分析，客户支持，机器人和自动驾驶，扩展了其强大的理解，推理和生成自然语言的能力。然而，基于LLM的应用程序的广泛部署暴露了关键的安全性和可靠性风险，例如恶意滥用、隐私泄露和服务中断的可能性，这些都会削弱用户信任并破坏社会安全。本文系统地概述了针对LLM和基于LLM的代理的对抗性攻击的细节。这些攻击在LLM中分为三个阶段：训练阶段攻击、推理阶段攻击和可用性和完整性攻击。对于每个阶段，我们都会分析代表性和最近引入的攻击方法及其相应防御的详细信息。我们希望我们的调查能够提供一个很好的教程和对LLM安全性的全面了解，尤其是对于LLM的攻击。我们希望引起人们对广泛部署的基于LLM的应用程序固有风险的关注，并强调迫切需要针对不断变化的威胁制定强有力的缓解策略。



## **16. BadNAVer: Exploring Jailbreak Attacks On Vision-and-Language Navigation**

BadNAVer：探索视觉和语言导航的越狱攻击 cs.RO

8 pages, 4 figures

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12443v1) [paper-pdf](http://arxiv.org/pdf/2505.12443v1)

**Authors**: Wenqi Lyu, Zerui Li, Yanyuan Qiao, Qi Wu

**Abstract**: Multimodal large language models (MLLMs) have recently gained attention for their generalization and reasoning capabilities in Vision-and-Language Navigation (VLN) tasks, leading to the rise of MLLM-driven navigators. However, MLLMs are vulnerable to jailbreak attacks, where crafted prompts bypass safety mechanisms and trigger undesired outputs. In embodied scenarios, such vulnerabilities pose greater risks: unlike plain text models that generate toxic content, embodied agents may interpret malicious instructions as executable commands, potentially leading to real-world harm. In this paper, we present the first systematic jailbreak attack paradigm targeting MLLM-driven navigator. We propose a three-tiered attack framework and construct malicious queries across four intent categories, concatenated with standard navigation instructions. In the Matterport3D simulator, we evaluate navigation agents powered by five MLLMs and report an average attack success rate over 90%. To test real-world feasibility, we replicate the attack on a physical robot. Our results show that even well-crafted prompts can induce harmful actions and intents in MLLMs, posing risks beyond toxic output and potentially leading to physical harm.

摘要: 多模式大型语言模型（MLLM）最近因其在视觉与语言导航（VLN）任务中的概括和推理能力而受到关注，导致MLLM驱动的导航器的兴起。然而，MLLM很容易受到越狱攻击，其中精心设计的提示绕过安全机制并触发不需要的输出。在具体场景中，此类漏洞带来了更大的风险：与生成有毒内容的纯文本模型不同，具体代理可能会将恶意指令解释为可执行命令，从而可能导致现实世界的伤害。在本文中，我们提出了第一个针对MLLM驱动导航器的系统越狱攻击范式。我们提出了一个三层攻击框架，并构建了四个意图类别的恶意查询，连接标准的导航指令。在Matterport3D模拟器中，我们评估了由五个MLLM驱动的导航代理，并报告了超过90%的平均攻击成功率。为了测试现实世界的可行性，我们复制了对物理机器人的攻击。我们的研究结果表明，即使是精心制作的提示也会在MLLM中引起有害的行为和意图，造成超出有毒输出的风险，并可能导致身体伤害。



## **17. IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems**

针对基于LLM的多代理系统的IP泄露攻击 cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12442v1) [paper-pdf](http://arxiv.org/pdf/2505.12442v1)

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.

摘要: 大型语言模型（LLM）的快速发展导致了通过协作执行复杂任务的多智能体系统（MAS）的出现。然而，MAS的复杂性质，包括其架构和代理交互，引发了有关知识产权（IP）保护的严重担忧。本文介绍MASLEAK，这是一种新型攻击框架，旨在从MAS应用程序中提取敏感信息。MASLEAK针对的是实用的黑匣子设置，其中对手不了解MAS架构或代理配置。对手只能通过其公共API与MAS交互，提交攻击查询$q$并观察最终代理的输出。受计算机蠕虫传播和感染脆弱网络主机的方式的启发，MASLEAK精心设计了对抗性查询$q$，以引发、传播和保留每个MAS代理的响应，这些响应揭示了全套专有组件，包括代理数量、系统布局、系统提示、任务指令和工具使用。我们构建了包含810个应用程序的第一个MAS应用程序合成数据集，并根据现实世界的MAS应用程序（包括Coze和CrewAI）评估MASLEAK。MASLEAK在提取MAS IP方面实现了高准确性，系统提示和任务指令的平均攻击成功率为87%，大多数情况下系统架构的平均攻击成功率为92%。最后，我们讨论了我们发现的影响和潜在的防御措施。



## **18. CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement**

捕获：上下文感知提示注入测试和稳健性增强 cs.CL

Accepted in ACL LLMSec Workshop 2025

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12368v1) [paper-pdf](http://arxiv.org/pdf/2505.12368v1)

**Authors**: Gauri Kholkar, Ratinder Ahuja

**Abstract**: Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations.

摘要: 提示注入仍然是大型语言模型的主要安全风险。然而，现有护栏模型在上下文感知环境中的功效仍然没有得到充分的探索，因为它们通常依赖于静态攻击基准。此外，他们还有过度防御的倾向。我们引入了CAPTURE，这是一种新型的上下文感知基准，通过最少的领域内示例来评估攻击检测和过度防御倾向。我们的实验表明，当前的即时注射护栏模型在对抗性情况下存在高假阴性，在良性情况下存在过多假阳性，凸显了严重的局限性。



## **19. The Tower of Babel Revisited: Multilingual Jailbreak Prompts on Closed-Source Large Language Models**

重访巴别塔：多语言越狱依赖闭源大型语言模型 cs.CL

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12287v1) [paper-pdf](http://arxiv.org/pdf/2505.12287v1)

**Authors**: Linghan Huang, Haolin Jin, Zhaoge Bi, Pengyue Yang, Peizhou Zhao, Taozhao Chen, Xiongfei Wu, Lei Ma, Huaming Chen

**Abstract**: Large language models (LLMs) have seen widespread applications across various domains, yet remain vulnerable to adversarial prompt injections. While most existing research on jailbreak attacks and hallucination phenomena has focused primarily on open-source models, we investigate the frontier of closed-source LLMs under multilingual attack scenarios. We present a first-of-its-kind integrated adversarial framework that leverages diverse attack techniques to systematically evaluate frontier proprietary solutions, including GPT-4o, DeepSeek-R1, Gemini-1.5-Pro, and Qwen-Max. Our evaluation spans six categories of security contents in both English and Chinese, generating 38,400 responses across 32 types of jailbreak attacks. Attack success rate (ASR) is utilized as the quantitative metric to assess performance from three dimensions: prompt design, model architecture, and language environment. Our findings suggest that Qwen-Max is the most vulnerable, while GPT-4o shows the strongest defense. Notably, prompts in Chinese consistently yield higher ASRs than their English counterparts, and our novel Two-Sides attack technique proves to be the most effective across all models. This work highlights a dire need for language-aware alignment and robust cross-lingual defenses in LLMs, and we hope it will inspire researchers, developers, and policymakers toward more robust and inclusive AI systems.

摘要: 大型语言模型（LLM）已经在各个领域得到了广泛的应用，但仍然容易受到对抗性提示注入的影响。虽然大多数关于越狱攻击和幻觉现象的现有研究主要集中在开源模型上，但我们研究了多语言攻击场景下闭源LLM的前沿。我们提出了一个首创的集成对抗框架，该框架利用各种攻击技术来系统地评估前沿专有解决方案，包括GPT-4 o，DeepSeek-R1，Gemini-1.5-Pro和Qwen-Max。我们的评估涵盖了中英文六类安全内容，针对32种越狱攻击生成了38，400个响应。攻击成功率（ASB）被用作量化指标，从三个维度评估性能：提示设计、模型架构和语言环境。我们的研究结果表明，Qwen-Max最脆弱，而GPT-4 o表现出最强的防御能力。值得注意的是，中文提示始终比英语提示产生更高的ASB，而且我们新颖的双面攻击技术被证明是所有模型中最有效的。这项工作凸显了LLM中对语言感知一致和强大的跨语言防御的迫切需求，我们希望它能够激励研究人员、开发人员和政策制定者开发更强大、更具包容性的人工智能系统。



## **20. `Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs**

“照我说的做，而不是照我做的做”：针对多模式LLM的越狱提示攻击的半自动方法 cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2502.00735v3) [paper-pdf](http://arxiv.org/pdf/2502.00735v3)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen, Kim-Kwang Raymond Choo

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the state-of-the-art multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. Further, to better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flanking Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios.

摘要: 大型语言模型（LLM）由于处理文本、音频、图像和视频等多种类型输入数据的能力不断增强，已在各个领域得到广泛应用。虽然LLM在理解和生成不同场景的上下文方面表现出出色的性能，但它们很容易受到基于预算的攻击（主要通过文本输入）。本文中，我们介绍了针对多模式LLM的第一个基于语音的越狱攻击，称为侧翼攻击，它可以同时处理针对多模式LLM的不同类型的输入。我们的工作受到单语语音驱动的大型语言模型的最新进展的推动，这些模型在LLM的传统基于文本的漏洞之外引入了新的攻击表面。为了调查这些风险，我们研究了最先进的多模式LLM，它们可以通过音频输入等不同类型的输入访问，重点关注对抗性提示如何绕过其防御机制。我们提出了一种新颖的策略，其中不允许的提示两侧是良性的、叙述驱动的提示。它集成在侧翼攻击中，该攻击试图将交互上下文人性化并通过虚构的环境执行攻击。此外，为了更好地评估攻击性能，我们提出了一个用于策略违规检测的半自动自我评估框架。我们证明了侧翼攻击能够操纵最先进的LLM生成不对齐和禁止的输出，在七种禁止的情况下，平均攻击成功率从0.67到0.93不等。



## **21. Self-Destructive Language Model**

自毁语言模型 cs.LG

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12186v1) [paper-pdf](http://arxiv.org/pdf/2505.12186v1)

**Authors**: Yuhui Wang, Rongyi Zhu, Ting Wang

**Abstract**: Harmful fine-tuning attacks pose a major threat to the security of large language models (LLMs), allowing adversaries to compromise safety guardrails with minimal harmful data. While existing defenses attempt to reinforce LLM alignment, they fail to address models' inherent "trainability" on harmful data, leaving them vulnerable to stronger attacks with increased learning rates or larger harmful datasets. To overcome this critical limitation, we introduce SEAM, a novel alignment-enhancing defense that transforms LLMs into self-destructive models with intrinsic resilience to misalignment attempts. Specifically, these models retain their capabilities for legitimate tasks while exhibiting substantial performance degradation when fine-tuned on harmful data. The protection is achieved through a novel loss function that couples the optimization trajectories of benign and harmful data, enhanced with adversarial gradient ascent to amplify the self-destructive effect. To enable practical training, we develop an efficient Hessian-free gradient estimate with theoretical error bounds. Extensive evaluation across LLMs and datasets demonstrates that SEAM creates a no-win situation for adversaries: the self-destructive models achieve state-of-the-art robustness against low-intensity attacks and undergo catastrophic performance collapse under high-intensity attacks, rendering them effectively unusable. (warning: this paper contains potentially harmful content generated by LLMs.)

摘要: 有害的微调攻击对大型语言模型（LLM）的安全性构成了重大威胁，使攻击者能够以最小的有害数据破坏安全护栏。虽然现有的防御措施试图加强LLM对齐，但它们未能解决模型在有害数据上固有的“可训练性”，使它们容易受到学习率提高或更大的有害数据集的更强攻击。为了克服这一关键限制，我们引入了SEAM，这是一种新的增强防御机制，它将LLM转换为具有内在弹性的自毁模型，以应对未对准尝试。具体来说，这些模型保留了合法任务的能力，同时在对有害数据进行微调时表现出显著的性能下降。这种保护是通过一种新的损失函数来实现的，该函数将良性和有害数据的优化轨迹结合起来，通过对抗性梯度上升来增强，以放大自毁效应。为了实现实际训练，我们开发了一个有效的Hessian自由梯度估计与理论误差界。对LLM和数据集的广泛评估表明，SEAM为对手创造了一个没有胜利的局面：自毁模型对低强度攻击具有最先进的鲁棒性，并在高强度攻击下经历灾难性的性能崩溃，使它们实际上无法使用。（警告：本文包含由LLM生成的潜在有害内容。



## **22. EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective**

EVALOOP：从自我一致性的角度评估LLM编程稳健性 cs.SE

19 pages, 11 figures

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12185v1) [paper-pdf](http://arxiv.org/pdf/2505.12185v1)

**Authors**: Sen Fang, Weiyuan Ding, Bowen Xu

**Abstract**: Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.

摘要: 评估大型语言模型（LLM）的编程能力对于它们在软件工程中的有效使用至关重要。然而，当前的评估主要衡量静态基准上生成的代码的准确性，忽视了编程任务期间模型稳健性的关键方面。虽然对抗性攻击提供了有关模型稳健性的见解，但它们的有效性有限，并且评估可能会受到限制。当前用于稳健性评估的对抗攻击方法会产生不一致的结果，难以在不同的LLM之间提供统一的评估。我们引入EVALOOP，这是一种新型评估框架，从自一致性的角度评估稳健性，即利用流行软件工程任务中固有的自然二重性，例如，代码生成和代码摘要。EVALOOP启动独立反馈循环：LLM生成输出（例如，代码）来自输入（例如，自然语言规范），然后使用生成的输出作为输入来产生新的输出（例如，将该代码总结为新规范）。EVALOOP重复该过程以评估每个循环中EVALOOP的有效性。这种循环策略本质上评估稳健性，而不依赖任何外部攻击设置，提供了一个统一的指标来评估LLM在编程中的稳健性。我们评估了16个著名的LLM（例如，GPT-4.1，O 4-mini）在EVALOOP上发现EVALOOP通常会在十个循环内导致pass@1性能绝对下降5.01%-19.31%。有趣的是，稳健性并不总是与初始性能一致（即，一次性查询）;例如，GPT-3.5-Turbo尽管初始代码生成优于DeepSeek-V2，但在重复评估循环中表现出较低的鲁棒性。



## **23. ImF: Implicit Fingerprint for Large Language Models**

ImF：大型语言模型的隐式指纹 cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2503.21805v2) [paper-pdf](http://arxiv.org/pdf/2503.21805v2)

**Authors**: Wu jiaxuan, Peng Wanli, Fu hang, Xue Yiming, Wen juan

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing fingerprinting techniques typically embed identifiable patterns with weak semantic coherence, resulting in fingerprints that significantly differ from the natural question-answering (QA) behavior inherent to LLMs. This discrepancy undermines the stealthiness of the embedded fingerprints and makes them vulnerable to adversarial attacks. In this paper, we first demonstrate the critical vulnerability of existing fingerprint embedding methods by introducing a novel adversarial attack named Generation Revision Intervention (GRI) attack. GRI attack exploits the semantic fragility of current fingerprinting methods, effectively erasing fingerprints by disrupting their weakly correlated semantic structures. Our empirical evaluation highlights that traditional fingerprinting approaches are significantly compromised by the GRI attack, revealing severe limitations in their robustness under realistic adversarial conditions. To advance the state-of-the-art in model fingerprinting, we propose a novel model fingerprint paradigm called Implicit Fingerprints (ImF). ImF leverages steganography techniques to subtly embed ownership information within natural texts, subsequently using Chain-of-Thought (CoT) prompting to construct semantically coherent and contextually natural QA pairs. This design ensures that fingerprints seamlessly integrate with the standard model behavior, remaining indistinguishable from regular outputs and substantially reducing the risk of accidental triggering and targeted removal. We conduct a comprehensive evaluation of ImF on 15 diverse LLMs, spanning different architectures and varying scales.

摘要: 训练大型语言模型（LLM）是资源密集型且昂贵的，因此保护LLM的知识产权（IP）至关重要。最近，将指纹嵌入LLM已成为建立模型所有权的流行方法。然而，现有的指纹识别技术通常嵌入具有弱语义一致性的可识别模式，导致指纹与LLM固有的自然问答（QA）行为显着不同。这种差异削弱了嵌入指纹的隐蔽性，并使它们容易受到对抗攻击。在本文中，我们首先证明了现有的指纹嵌入方法的关键漏洞，通过引入一种新的对抗性攻击，称为生成修订干预（GRI）攻击。GRI攻击利用了当前指纹识别方法的语义脆弱性，通过破坏指纹的弱相关语义结构来有效地擦除指纹。我们的经验评估强调，传统的指纹识别方法受到GRI攻击的严重影响，在现实的对抗条件下，它们的鲁棒性存在严重的局限性。为了推进国家的最先进的模型指纹，我们提出了一种新的模型指纹范例称为隐式指纹（ImF）。ImF利用隐写技术将所有权信息巧妙地嵌入自然文本中，随后使用思想链（CoT）提示构建语义连贯且上下文自然的QA对。这种设计确保指纹与标准模型行为无缝集成，与常规输出保持无区别，并大幅降低意外触发和有针对性删除的风险。我们对15个不同的LLM进行了ImF的全面评估，涵盖不同的架构和不同的规模。



## **24. Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement**

为什么不按照你所知道的去做呢？通过自我意识防护增强释放LLM的安全潜力 cs.CL

Acccepted by ACL 2025 Findings, 21 pages, 9 figures, 14 tables

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.12060v1) [paper-pdf](http://arxiv.org/pdf/2505.12060v1)

**Authors**: Peng Ding, Jun Kuang, Zongyu Wang, Xuezhi Cao, Xunliang Cai, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities across various tasks but remain vulnerable to meticulously crafted jailbreak attacks. In this paper, we identify a critical safety gap: while LLMs are adept at detecting jailbreak prompts, they often produce unsafe responses when directly processing these inputs. Inspired by this insight, we propose SAGE (Self-Aware Guard Enhancement), a training-free defense strategy designed to align LLMs' strong safety discrimination performance with their relatively weaker safety generation ability. SAGE consists of two core components: a Discriminative Analysis Module and a Discriminative Response Module, enhancing resilience against sophisticated jailbreak attempts through flexible safety discrimination instructions. Extensive experiments demonstrate SAGE's effectiveness and robustness across various open-source and closed-source LLMs of different sizes and architectures, achieving an average 99% defense success rate against numerous complex and covert jailbreak methods while maintaining helpfulness on general benchmarks. We further conduct mechanistic interpretability analysis through hidden states and attention distributions, revealing the underlying mechanisms of this detection-generation discrepancy. Our work thus contributes to developing future LLMs with coherent safety awareness and generation behavior. Our code and datasets are publicly available at https://github.com/NJUNLP/SAGE.

摘要: 大型语言模型（LLM）在各种任务中表现出令人印象深刻的能力，但仍然容易受到精心设计的越狱攻击。在本文中，我们发现了一个关键的安全差距：虽然LLM善于检测越狱提示，但在直接处理这些输入时，它们通常会产生不安全的响应。受这一见解的启发，我们提出了SAGE（Self-Aware Guard Enhancement），这是一种免训练的防御策略，旨在将LLM强大的安全区分性能与其相对较弱的安全生成能力保持一致。SAGE由两个核心组件组成：区分分析模块和区分响应模块，通过灵活的安全区分指令增强针对复杂越狱企图的弹性。大量的实验证明了SAGE在不同规模和架构的各种开源和闭源LLM上的有效性和鲁棒性，对许多复杂和隐蔽的越狱方法实现了平均99%的防御成功率，同时在一般基准测试中保持了有用性。我们进一步通过隐藏状态和注意力分布进行机械可解释性分析，揭示了这种检测生成差异的潜在机制。因此，我们的工作有助于开发具有连贯安全意识和发电行为的未来LLM。我们的代码和数据集可在https://github.com/NJUNLP/SAGE上公开获取。



## **25. FIGhost: Fluorescent Ink-based Stealthy and Flexible Backdoor Attacks on Physical Traffic Sign Recognition**

FIGhost：基于荧光墨水的隐形和灵活后门攻击物理交通标志识别 cs.CV

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.12045v1) [paper-pdf](http://arxiv.org/pdf/2505.12045v1)

**Authors**: Shuai Yuan, Guowen Xu, Hongwei Li, Rui Zhang, Xinyuan Qian, Wenbo Jiang, Hangcheng Cao, Qingchuan Zhao

**Abstract**: Traffic sign recognition (TSR) systems are crucial for autonomous driving but are vulnerable to backdoor attacks. Existing physical backdoor attacks either lack stealth, provide inflexible attack control, or ignore emerging Vision-Large-Language-Models (VLMs). In this paper, we introduce FIGhost, the first physical-world backdoor attack leveraging fluorescent ink as triggers. Fluorescent triggers are invisible under normal conditions and activated stealthily by ultraviolet light, providing superior stealthiness, flexibility, and untraceability. Inspired by real-world graffiti, we derive realistic trigger shapes and enhance their robustness via an interpolation-based fluorescence simulation algorithm. Furthermore, we develop an automated backdoor sample generation method to support three attack objectives. Extensive evaluations in the physical world demonstrate FIGhost's effectiveness against state-of-the-art detectors and VLMs, maintaining robustness under environmental variations and effectively evading existing defenses.

摘要: 交通标志识别（TSR）系统对于自动驾驶至关重要，但容易受到后门攻击。现有的物理后门攻击要么缺乏隐蔽性，提供不灵活的攻击控制，要么忽视新兴的视觉大数据模型（VLM）。在本文中，我们介绍了FIGhost，第一个利用荧光墨水作为触发器的物理世界后门攻击。荧光触发器在正常情况下是不可见的，并通过紫外线悄悄地激活，提供卓越的隐蔽性，灵活性和不可追溯性。受现实世界涂鸦的启发，我们推导出逼真的触发形状，并通过基于插值的荧光模拟算法增强其稳健性。此外，我们开发了一种自动后门样本生成方法来支持三个攻击目标。物理世界中的广泛评估证明了FIGGhost对最先进的探测器和VLM的有效性，在环境变化下保持稳健性并有效规避现有防御。



## **26. Vulnerability of Text-to-Image Models to Prompt Template Stealing: A Differential Evolution Approach**

文本到图像模型提示模板窃取的脆弱性：差异进化方法 cs.CL

14 pages,8 figures,4 tables

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2502.14285v2) [paper-pdf](http://arxiv.org/pdf/2502.14285v2)

**Authors**: Yurong Wu, Fangwen Mu, Qiuhong Zhang, Jinjing Zhao, Xinrun Xu, Lingrui Mei, Yang Wu, Lin Shi, Junjie Wang, Zhiming Ding, Yiwei Wang

**Abstract**: Prompt trading has emerged as a significant intellectual property concern in recent years, where vendors entice users by showcasing sample images before selling prompt templates that can generate similar images. This work investigates a critical security vulnerability: attackers can steal prompt templates using only a limited number of sample images. To investigate this threat, we introduce Prism, a prompt-stealing benchmark consisting of 50 templates and 450 images, organized into Easy and Hard difficulty levels. To identify the vulnerabity of VLMs to prompt stealing, we propose EvoStealer, a novel template stealing method that operates without model fine-tuning by leveraging differential evolution algorithms. The system first initializes population sets using multimodal large language models (MLLMs) based on predefined patterns, then iteratively generates enhanced offspring through MLLMs. During evolution, EvoStealer identifies common features across offspring to derive generalized templates. Our comprehensive evaluation conducted across open-source (INTERNVL2-26B) and closed-source models (GPT-4o and GPT-4o-mini) demonstrates that EvoStealer's stolen templates can reproduce images highly similar to originals and effectively generalize to other subjects, significantly outperforming baseline methods with an average improvement of over 10%. Moreover, our cost analysis reveals that EvoStealer achieves template stealing with negligible computational expenses. Our code and dataset are available at https://github.com/whitepagewu/evostealer.

摘要: 近年来，提示交易已成为一个重要的知识产权问题，供应商在出售可以生成类似图像的提示模板之前先展示样本图像来吸引用户。这项工作调查了一个关键的安全漏洞：攻击者可以仅使用有限数量的样本图像窃取提示模板。为了调查这种威胁，我们引入了Prism，这是一个预算窃取基准，由50个模板和450个图像组成，按Easy和Hard难度级别组织。为了确定VLM是否可能促使窃取，我们提出了EvoStealer，这是一种新颖的模板窃取方法，通过利用差异进化算法无需模型微调即可运行。该系统首先使用基于预定义模式的多模式大型语言模型（MLLM）来初始化种群集，然后通过MLLM迭代生成增强的后代。在进化过程中，EvoStealer识别后代的共同特征以推导出广义模板。我们对开源（INTENVL 2 - 26 B）和闭源模型（GPT-4 o和GPT-4 o-mini）进行的全面评估表明，EvoStealer的被盗模板可以复制与原始图像高度相似的图像，并有效地推广到其他主题，显着优于基线方法，平均改进超过10%。此外，我们的成本分析表明，EvoStealer只需微不足道的计算费用即可实现模板窃取。我们的代码和数据集可以在https://github.com/whitepagewu/evostealer上找到。



## **27. Understanding and Enhancing the Transferability of Jailbreaking Attacks**

了解并增强越狱攻击的可转移性 cs.LG

Accepted by ICLR 2025

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2502.03052v2) [paper-pdf](http://arxiv.org/pdf/2502.03052v2)

**Authors**: Runqi Lin, Bo Han, Fengwang Li, Tongling Liu

**Abstract**: Jailbreaking attacks can effectively manipulate open-source large language models (LLMs) to produce harmful responses. However, these attacks exhibit limited transferability, failing to disrupt proprietary LLMs consistently. To reliably identify vulnerabilities in proprietary LLMs, this work investigates the transferability of jailbreaking attacks by analysing their impact on the model's intent perception. By incorporating adversarial sequences, these attacks can redirect the source LLM's focus away from malicious-intent tokens in the original input, thereby obstructing the model's intent recognition and eliciting harmful responses. Nevertheless, these adversarial sequences fail to mislead the target LLM's intent perception, allowing the target LLM to refocus on malicious-intent tokens and abstain from responding. Our analysis further reveals the inherent distributional dependency within the generated adversarial sequences, whose effectiveness stems from overfitting the source LLM's parameters, resulting in limited transferability to target LLMs. To this end, we propose the Perceived-importance Flatten (PiF) method, which uniformly disperses the model's focus across neutral-intent tokens in the original input, thus obscuring malicious-intent tokens without relying on overfitted adversarial sequences. Extensive experiments demonstrate that PiF provides an effective and efficient red-teaming evaluation for proprietary LLMs.

摘要: 越狱攻击可以有效地操纵开源大型语言模型（LLM）以产生有害响应。然而，这些攻击的可转让性有限，无法一致破坏专有LLM。为了可靠地识别专有LLM中的漏洞，这项工作通过分析越狱攻击对模型意图感知的影响来研究越狱攻击的可转移性。通过结合对抗序列，这些攻击可以将源LLM的焦点从原始输入中的恶意意图标记重新定向，从而阻碍模型的意图识别并引发有害响应。然而，这些对抗序列未能误导目标LLM的意图感知，从而允许目标LLM重新关注恶意意图代币并放弃回应。我们的分析进一步揭示了生成的对抗序列内固有的分布依赖性，其有效性源于过度匹配源LLM的参数，导致目标LLM的可移植性有限。为此，我们提出了感知重要性拉平（PiF）方法，该方法将模型的焦点均匀分散到原始输入中的中立意图标记上，从而在不依赖过度匹配的对抗序列的情况下模糊恶意意图标记。大量实验表明，PiF为专有LLM提供了有效且高效的红色团队评估。



## **28. Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs**

Video-SafetyBench：视频LVLM安全评估的基准 cs.CV

Project page:  https://liuxuannan.github.io/Video-SafetyBench.github.io/

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.11842v1) [paper-pdf](http://arxiv.org/pdf/2505.11842v1)

**Authors**: Xuannan Liu, Zekun Li, Zheqi He, Peipei Li, Shuhan Xia, Xing Cui, Huaibo Huang, Xi Yang, Ran He

**Abstract**: The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies.

摘要: 大型视觉语言模型（LVLM）的越来越多的部署引发了潜在恶意输入下的安全问题。然而，现有的多模式安全评估主要关注静态图像输入暴露的模型漏洞，而忽略了可能引发明显安全风险的视频的时间动态。为了弥合这一差距，我们引入了Video-SafetyBench，这是第一个旨在评估LVLM在视频文本攻击下的安全性的综合基准。它由2，264个视频-文本对组成，涵盖48个细粒度的不安全类别，每个将合成视频与包含明显恶意的有害查询或良性查询配对，良性查询看似无害，但在与视频一起解释时会触发有害行为。为了生成语义准确的视频以进行安全评估，我们设计了一个可控的管道，将视频语义分解为主题图像（显示的内容）和运动文本（它如何移动），共同指导查询相关视频的合成。为了有效地评估不确定或边缘有害输出，我们提出了RJScore，这是一种新型的基于LLM的指标，它结合了判断模型的置信度和人类一致的决策阈值校准。大量实验表明，良性查询视频合成的平均攻击成功率为67.2%，揭示了视频诱导攻击的一致漏洞。我们相信Video-SafetyBench将促进未来对基于视频的安全评估和防御策略的研究。



## **29. On Membership Inference Attacks in Knowledge Distillation**

知识提炼中的成员推理攻击 cs.LG

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.11837v1) [paper-pdf](http://arxiv.org/pdf/2505.11837v1)

**Authors**: Ziyao Cui, Minxing Zhang, Jian Pei

**Abstract**: Nowadays, Large Language Models (LLMs) are trained on huge datasets, some including sensitive information. This poses a serious privacy concern because privacy attacks such as Membership Inference Attacks (MIAs) may detect this sensitive information. While knowledge distillation compresses LLMs into efficient, smaller student models, its impact on privacy remains underexplored. In this paper, we investigate how knowledge distillation affects model robustness against MIA. We focus on two questions. First, how is private data protected in teacher and student models? Second, how can we strengthen privacy preservation against MIAs in knowledge distillation? Through comprehensive experiments, we show that while teacher and student models achieve similar overall MIA accuracy, teacher models better protect member data, the primary target of MIA, whereas student models better protect non-member data. To address this vulnerability in student models, we propose 5 privacy-preserving distillation methods and demonstrate that they successfully reduce student models' vulnerability to MIA, with ensembling further stabilizing the robustness, offering a reliable approach for distilling more secure and efficient student models. Our implementation source code is available at https://github.com/richardcui18/MIA_in_KD.

摘要: 如今，大型语言模型（LLM）是在巨大的数据集上训练的，其中一些数据集包括敏感信息。这会带来严重的隐私问题，因为会员推断攻击（MIA）等隐私攻击可能会检测到此敏感信息。虽然知识提炼将LLM压缩为高效、更小的学生模型，但其对隐私的影响仍然没有得到充分研究。本文中，我们研究知识提炼如何影响模型针对MIA的鲁棒性。我们重点关注两个问题。首先，教师和学生模型中如何保护私人数据？其次，如何在知识蒸馏中加强针对MIA的隐私保护？通过全面的实验，我们表明，虽然教师模型和学生模型实现了相似的整体MIA准确性，但教师模型更好地保护了成员数据（MIA的主要目标），而学生模型更好地保护了非成员数据。为了解决学生模型中的这一漏洞，我们提出了5种保护隐私的提炼方法，并证明它们可以成功地降低学生模型对MIA的脆弱性，进一步稳定稳健性，为提炼更安全、更高效的学生模型提供了可靠的方法。我们的实现源代码可在https://github.com/richardcui18/MIA_in_KD上获取。



## **30. Multilingual Collaborative Defense for Large Language Models**

大型语言模型的多语言协作防御 cs.CL

19 pages, 4figures

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.11835v1) [paper-pdf](http://arxiv.org/pdf/2505.11835v1)

**Authors**: Hongliang Li, Jinan Xu, Gengping Cui, Changhao Guan, Fengran Mo, Kaiyu Huang

**Abstract**: The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at https://github.com/HLiang-Lee/MCD.

摘要: 大型语言模型（LLM）的稳健性和安全性已成为一个重要的研究领域。一个值得注意的漏洞是，通过将有害查询翻译成罕见或代表性不足的语言来绕过LLM保障措施，这是“越狱”这些模型的一种简单而有效的方法。尽管人们的担忧日益加剧，但针对多语言场景下LLM保护的研究有限，凸显了加强多语言安全的迫切需要。在这项工作中，我们调查了不同语言中的各种攻击特征之间的相关性，并提出了多语言协作防御（MCB），这是一种新型学习方法，可以自动优化连续的软安全提示，以促进LLM的多语言保护。MCB方法具有三个优点：首先，它有效地提高了跨多种语言的性能保护。其次，MCB保持了强大的概括能力，同时最大限度地降低了错误拒绝率。第三，MCB缓解了LLM培训库失衡造成的语言安全失调。为了评估MCB的有效性，我们手动构建常用越狱基准（例如MaliciousDirecct和AdvBench）的多语言版本，以评估各种保障方法。此外，我们以未充分代表（零镜头）语言引入这些数据集，以验证MCB的语言可移植性。结果表明，MCB在防范多语言越狱企图方面优于现有方法，同时还表现出强大的语言传输能力。我们的代码可在https://github.com/HLiang-Lee/MCD上获取。



## **31. Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models**

基于通用魔法词的LLM越狱安全机制 cs.CL

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2501.18280v3) [paper-pdf](http://arxiv.org/pdf/2501.18280v3)

**Authors**: Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang

**Abstract**: The security issue of large language models (LLMs) has gained wide attention recently, with various defense mechanisms developed to prevent harmful output, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the output distribution of text embedding models is severely biased with a large mean. Inspired by this observation, we propose novel, efficient methods to search for **universal magic words** that attack text embedding models. Universal magic words as suffixes can shift the embedding of any text towards the bias direction, thus manipulating the similarity of any text pair and misleading safeguards. Attackers can jailbreak the safeguards by appending magic words to user prompts and requiring LLMs to end answers with magic words. Experiments show that magic word attacks significantly degrade safeguard performance on JailbreakBench, cause real-world chatbots to produce harmful outputs in full-pipeline attacks, and generalize across input/output texts, models, and languages. To eradicate this security risk, we also propose defense methods against such attacks, which can correct the bias of text embeddings and improve downstream performance in a train-free manner.

摘要: 大型语言模型（LLM）的安全问题最近受到广泛关注，各种防御机制被开发出来来防止有害输出，其中基于文本嵌入模型的保护措施是基本防御。通过测试，我们发现文本嵌入模型的输出分布存在严重偏差，平均值很大。受这一观察的启发，我们提出了新颖、有效的方法来搜索攻击文本嵌入模型的 ** 通用魔法词 **。作为后缀的通用神奇词可以将任何文本的嵌入转向偏向方向，从而操纵任何文本对的相似性并产生误导性的保障措施。攻击者可以通过在用户提示中添加魔法词并要求LLM以魔法词结束回答来越狱保护措施。实验表明，魔法词攻击会显着降低JailbreakBench上的防护性能，导致现实世界的聊天机器人在全管道攻击中产生有害输出，并在输入/输出文本、模型和语言上进行概括。为了消除这种安全风险，我们还提出了针对此类攻击的防御方法，可以纠正文本嵌入的偏见，并以无训练的方式提高下游性能。



## **32. Efficient Indirect LLM Jailbreak via Multimodal-LLM Jailbreak**

通过多模式LLM越狱高效间接LLM越狱 cs.AI

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2405.20015v2) [paper-pdf](http://arxiv.org/pdf/2405.20015v2)

**Authors**: Zhenxing Niu, Yuyao Sun, Haoxuan Ji, Zheng Lin, Haichang Gao, Xinbo Gao, Gang Hua, Rong Jin

**Abstract**: This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreak methods that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) built upon the target LLM. Subsequently, we perform an efficient MLLM jailbreak and obtain a jailbreaking embedding. Finally, we convert the embedding into a textual jailbreaking suffix to carry out the jailbreak of target LLM. Compared to the direct LLM-jailbreak methods, our indirect jailbreaking approach is more efficient, as MLLMs are more vulnerable to jailbreak than pure LLM. Additionally, to improve the attack success rate of jailbreak, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art jailbreak methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class generalization abilities.

摘要: 本文重点关注针对大型语言模型（LLM）的越狱攻击，促使它们生成令人反感的内容来响应有害的用户查询。与之前直接面向LLM的LLM越狱方法不同，我们的方法首先构建基于目标LLM的多模式大型语言模型（MLLM）。随后，我们执行高效的MLLM越狱并获得越狱嵌入。最后，我们将嵌入转换为文本越狱后缀来执行目标LLM的越狱。与直接LLM越狱方法相比，我们的间接越狱方法更有效，因为MLLM比纯粹的LLM更容易受到越狱的影响。此外，为了提高越狱的攻击成功率，我们提出了一种图像-文本语义匹配方案来识别合适的初始输入。大量实验表明，我们的方法在效率和有效性方面都超过了当前最先进的越狱方法。此外，我们的方法具有出色的跨类概括能力。



## **33. Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey**

过去10年视觉任务的对抗性攻击：一项调查 cs.CV

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2410.23687v2) [paper-pdf](http://arxiv.org/pdf/2410.23687v2)

**Authors**: Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Zhe Liu

**Abstract**: With the advent of Large Vision-Language Models (LVLMs), new attack vectors, such as cognitive bias, prompt injection, and jailbreaking, have emerged. Understanding these attacks promotes system robustness improvement and neural networks demystification. However, existing surveys often target attack taxonomy and lack in-depth analysis like 1) unified insights into adversariality, transferability, and generalization; 2) detailed evaluations framework; 3) motivation-driven attack categorizations; and 4) an integrated perspective on both traditional and LVLM attacks. This article addresses these gaps by offering a thorough summary of traditional and LVLM adversarial attacks, emphasizing their connections and distinctions, and providing actionable insights for future research.

摘要: 随着大型视觉语言模型（LVLM）的出现，出现了新的攻击向量，如认知偏差，即时注入和越狱。了解这些攻击有助于提高系统的鲁棒性和神经网络的神秘性。然而，现有的调查通常针对攻击分类，缺乏深入的分析，如1）对对抗性，可转移性和泛化的统一见解; 2）详细的评估框架; 3）动机驱动的攻击分类;以及4）对传统和LVLM攻击的综合观点。本文通过全面总结传统和LVLM对抗性攻击来解决这些差距，强调它们的联系和区别，并为未来的研究提供可操作的见解。



## **34. JULI: Jailbreak Large Language Models by Self-Introspection**

JULI：通过自我反省越狱大型语言模型 cs.LG

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.11790v1) [paper-pdf](http://arxiv.org/pdf/2505.11790v1)

**Authors**: Jesson Wang, Zhanhao Hu, David Wagner

**Abstract**: Large Language Models (LLMs) are trained with safety alignment to prevent generating malicious content. Although some attacks have highlighted vulnerabilities in these safety-aligned LLMs, they typically have limitations, such as necessitating access to the model weights or the generation process. Since proprietary models through API-calling do not grant users such permissions, these attacks find it challenging to compromise them. In this paper, we propose Jailbreaking Using LLM Introspection (JULI), which jailbreaks LLMs by manipulating the token log probabilities, using a tiny plug-in block, BiasNet. JULI relies solely on the knowledge of the target LLM's predicted token log probabilities. It can effectively jailbreak API-calling LLMs under a black-box setting and knowing only top-$5$ token log probabilities. Our approach demonstrates superior effectiveness, outperforming existing state-of-the-art (SOTA) approaches across multiple metrics.

摘要: 大型语言模型（LLM）经过安全调整训练，以防止生成恶意内容。尽管一些攻击凸显了这些安全一致的LLM中的漏洞，但它们通常具有局限性，例如需要访问模型权重或生成过程。由于通过API调用的专有模型不会向用户授予此类权限，因此这些攻击发现很难损害它们。在本文中，我们提出了使用LLM内省越狱（JULI），它通过使用一个微型插件块BiasNet操纵令牌日志概率来越狱LLM。JULI仅依赖于目标LLM预测的令牌日志概率的知识。它可以在黑匣子设置下有效越狱API调用LLM，并且仅知道最高5美元的代币日志概率。我们的方法表现出卓越的有效性，在多个指标上优于现有的最先进（SOTA）方法。



## **35. EnvInjection: Environmental Prompt Injection Attack to Multi-modal Web Agents**

EnvInjecting：对多模式Web代理的环境提示注入攻击 cs.LG

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11717v1) [paper-pdf](http://arxiv.org/pdf/2505.11717v1)

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. Environmental prompt injection attacks manipulate the environment to induce the web agent to perform a specific, attacker-chosen action--referred to as the target action. However, existing attacks suffer from limited effectiveness or stealthiness, or are impractical in real-world settings. In this work, we propose EnvInjection, a new attack that addresses these limitations. Our attack adds a perturbation to the raw pixel values of the rendered webpage, which can be implemented by modifying the webpage's source code. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the target action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple webpage datasets shows that EnvInjection is highly effective and significantly outperforms existing baselines.

摘要: 基于多模态大语言模型（MLLM）的Web代理通过基于网页的屏幕截图生成动作来与网页环境交互。环境提示注入攻击操纵环境，诱导Web代理执行特定的攻击者选择的操作-称为目标操作。然而，现有的攻击遭受有限的有效性或隐蔽性，或者在现实世界中是不切实际的。在这项工作中，我们提出了EnvInjection，一种新的攻击，解决了这些限制。我们的攻击对渲染网页的原始像素值添加了扰动，这可以通过修改网页的源代码来实现。这些受干扰的像素被映射到屏幕截图后，受干扰会导致Web代理执行目标操作。我们将寻找扰动的任务定义为优化问题。解决这个问题的一个关键挑战是原始像素值和屏幕截图之间的映射是不可微的，因此很难将梯度反向传播到扰动。为了克服这个问题，我们训练神经网络来逼近映射，并应用投影梯度下降来解决重新制定的优化问题。对多个网页数据集的广泛评估表明，EnvInjecting非常有效，并且显着优于现有基线。



## **36. To Think or Not to Think: Exploring the Unthinking Vulnerability in Large Reasoning Models**

思考或不思考：探索大型推理模型中不思考的脆弱性 cs.CL

39 pages, 13 tables, 14 figures

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2502.12202v2) [paper-pdf](http://arxiv.org/pdf/2502.12202v2)

**Authors**: Zihao Zhu, Hongbao Zhang, Ruotong Wang, Ke Xu, Siwei Lyu, Baoyuan Wu

**Abstract**: Large Reasoning Models (LRMs) are designed to solve complex tasks by generating explicit reasoning traces before producing final answers. However, we reveal a critical vulnerability in LRMs -- termed Unthinking Vulnerability -- wherein the thinking process can be bypassed by manipulating special delimiter tokens. It is empirically demonstrated to be widespread across mainstream LRMs, posing both a significant risk and potential utility, depending on how it is exploited. In this paper, we systematically investigate this vulnerability from both malicious and beneficial perspectives. On the malicious side, we introduce Breaking of Thought (BoT), a novel attack that enables adversaries to bypass the thinking process of LRMs, thereby compromising their reliability and availability. We present two variants of BoT: a training-based version that injects backdoor during the fine-tuning stage, and a training-free version based on adversarial attack during the inference stage. As a potential defense, we propose thinking recovery alignment to partially mitigate the vulnerability. On the beneficial side, we introduce Monitoring of Thought (MoT), a plug-and-play framework that allows model owners to enhance efficiency and safety. It is implemented by leveraging the same vulnerability to dynamically terminate redundant or risky reasoning through external monitoring. Extensive experiments show that BoT poses a significant threat to reasoning reliability, while MoT provides a practical solution for preventing overthinking and jailbreaking. Our findings expose an inherent flaw in current LRM architectures and underscore the need for more robust reasoning systems in the future.

摘要: 大型推理模型（LRM）旨在通过在生成最终答案之前生成显式推理痕迹来解决复杂任务。然而，我们揭示了LRM中的一个关键漏洞--称为“无思考漏洞”--其中思维过程可以通过操纵特殊的Inbox令牌来绕过。经验证明，它在主流LRM中广泛存在，既构成重大风险，又构成潜在效用，具体取决于它的利用方式。在本文中，我们从恶意和有益的角度系统地调查了该漏洞。在恶意方面，我们引入了突破思想（BoT），这是一种新型攻击，使对手能够绕过LRM的思维过程，从而损害其可靠性和可用性。我们提出了BoT的两个变体：一个是在微调阶段注入后门的基于训练的版本，另一个是在推理阶段基于对抗攻击的免训练版本。作为一种潜在的防御措施，我们建议考虑恢复对齐来部分缓解漏洞。从有利的方面来说，我们引入了思维监控（MoT），这是一种即插即用框架，允许模型所有者提高效率和安全性。它是通过利用相同的漏洞通过外部监控动态终止冗余或有风险的推理来实现的。大量实验表明，BoT对推理可靠性构成了重大威胁，而MoT则为防止过度思考和越狱提供了实用的解决方案。我们的研究结果暴露了当前LRM架构中的固有缺陷，并强调了未来对更强大推理系统的需求。



## **37. ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks**

ProxyPrompt：保护系统安全免受即时提取攻击 cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11459v1) [paper-pdf](http://arxiv.org/pdf/2505.11459v1)

**Authors**: Zhixiong Zhuang, Maria-Irina Nicolae, Hui-Po Wang, Mario Fritz

**Abstract**: The integration of large language models (LLMs) into a wide range of applications has highlighted the critical role of well-crafted system prompts, which require extensive testing and domain expertise. These prompts enhance task performance but may also encode sensitive information and filtering criteria, posing security risks if exposed. Recent research shows that system prompts are vulnerable to extraction attacks, while existing defenses are either easily bypassed or require constant updates to address new threats. In this work, we introduce ProxyPrompt, a novel defense mechanism that prevents prompt leakage by replacing the original prompt with a proxy. This proxy maintains the original task's utility while obfuscating the extracted prompt, ensuring attackers cannot reproduce the task or access sensitive information. Comprehensive evaluations on 264 LLM and system prompt pairs show that ProxyPrompt protects 94.70% of prompts from extraction attacks, outperforming the next-best defense, which only achieves 42.80%.

摘要: 将大型语言模型（LLM）集成到广泛的应用程序中凸显了精心设计的系统提示的关键作用，这需要广泛的测试和领域专业知识。这些提示增强了任务性能，但也可能对敏感信息和过滤标准进行编码，如果暴露，就会带来安全风险。最近的研究表明，系统提示很容易受到提取攻击，而现有的防御系统要么很容易被绕过，要么需要不断更新来应对新威胁。在这项工作中，我们引入了ProxyPrompt，这是一种新颖的防御机制，通过用代理替换原始提示来防止提示泄露。该代理维护原始任务的实用程序，同时模糊提取的提示，确保攻击者无法复制任务或访问敏感信息。对264个LLM和系统提示对的综合评估表明，ProxyPrompt可以保护94.70%的提示免受提取攻击，优于次佳防御（次佳防御仅达到42.80%）。



## **38. LLMs unlock new paths to monetizing exploits**

LLM开辟了利用货币化的新途径 cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11449v1) [paper-pdf](http://arxiv.org/pdf/2505.11449v1)

**Authors**: Nicholas Carlini, Milad Nasr, Edoardo Debenedetti, Barry Wang, Christopher A. Choquette-Choo, Daphne Ippolito, Florian Tramèr, Matthew Jagielski

**Abstract**: We argue that Large language models (LLMs) will soon alter the economics of cyberattacks. Instead of attacking the most commonly used software and monetizing exploits by targeting the lowest common denominator among victims, LLMs enable adversaries to launch tailored attacks on a user-by-user basis. On the exploitation front, instead of human attackers manually searching for one difficult-to-identify bug in a product with millions of users, LLMs can find thousands of easy-to-identify bugs in products with thousands of users. And on the monetization front, instead of generic ransomware that always performs the same attack (encrypt all your data and request payment to decrypt), an LLM-driven ransomware attack could tailor the ransom demand based on the particular content of each exploited device.   We show that these two attacks (and several others) are imminently practical using state-of-the-art LLMs. For example, we show that without any human intervention, an LLM finds highly sensitive personal information in the Enron email dataset (e.g., an executive having an affair with another employee) that could be used for blackmail. While some of our attacks are still too expensive to scale widely today, the incentives to implement these attacks will only increase as LLMs get cheaper. Thus, we argue that LLMs create a need for new defense-in-depth approaches.

摘要: 我们认为大型语言模型（LLM）很快就会改变网络攻击的经济学。LLM不是攻击最常用的软件并通过针对受害者中最低的共同点来利用漏洞获利，而是使对手能够针对每个用户发起量身定制的攻击。在剥削方面，LLM可以在拥有数千名用户的产品中找到数千个易于识别的错误，而不是人类攻击者手动搜索拥有数百万用户的产品中的数千个易于识别的错误。在货币化方面，LLM驱动的勒索软件攻击不是总是执行相同攻击（加密所有数据并请求付费解密）的通用勒索软件，而是可以根据每个被利用设备的特定内容定制赎金需求。   我们表明，使用最先进的LLM，这两种攻击（以及其他几种攻击）迫在眉睫。例如，我们表明，在没有任何人为干预的情况下，LLM可以在安然电子邮件数据集中找到高度敏感的个人信息（例如，高管与另一名员工有外遇）可能用于勒索。虽然今天我们的一些攻击仍然过于昂贵，无法广泛扩展，但随着LLM变得更便宜，实施这些攻击的动机只会增加。因此，我们认为LLM需要新的深度防御方法。



## **39. CARES: Comprehensive Evaluation of Safety and Adversarial Robustness in Medical LLMs**

CARES：医学LLM安全性和对抗稳健性的综合评估 cs.CL

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11413v1) [paper-pdf](http://arxiv.org/pdf/2505.11413v1)

**Authors**: Sijia Chen, Xiaomin Li, Mengxue Zhang, Eric Hanchen Jiang, Qingcheng Zeng, Chen-Hsiang Yu

**Abstract**: Large language models (LLMs) are increasingly deployed in medical contexts, raising critical concerns about safety, alignment, and susceptibility to adversarial manipulation. While prior benchmarks assess model refusal capabilities for harmful prompts, they often lack clinical specificity, graded harmfulness levels, and coverage of jailbreak-style attacks. We introduce CARES (Clinical Adversarial Robustness and Evaluation of Safety), a benchmark for evaluating LLM safety in healthcare. CARES includes over 18,000 prompts spanning eight medical safety principles, four harm levels, and four prompting styles: direct, indirect, obfuscated, and role-play, to simulate both malicious and benign use cases. We propose a three-way response evaluation protocol (Accept, Caution, Refuse) and a fine-grained Safety Score metric to assess model behavior. Our analysis reveals that many state-of-the-art LLMs remain vulnerable to jailbreaks that subtly rephrase harmful prompts, while also over-refusing safe but atypically phrased queries. Finally, we propose a mitigation strategy using a lightweight classifier to detect jailbreak attempts and steer models toward safer behavior via reminder-based conditioning. CARES provides a rigorous framework for testing and improving medical LLM safety under adversarial and ambiguous conditions.

摘要: 大型语言模型（LLM）越来越多地被部署在医疗环境中，引起了人们对安全性、对齐性和对抗性操纵敏感性的严重关注。虽然先前的基准评估模型拒绝有害提示的能力，但它们通常缺乏临床特异性，分级的危害级别和越狱式攻击的覆盖范围。我们介绍了CARES（临床对抗性鲁棒性和安全性评估），这是评估LLM在医疗保健中安全性的基准。CARES包含超过18，000个提示，涵盖八个医疗安全原则，四个危害级别和四种提示风格：直接，间接，模糊和角色扮演，以模拟恶意和良性用例。我们提出了一个三向响应评估协议（接受、谨慎、拒绝）和细粒度的安全评分指标来评估模型行为。我们的分析表明，许多最先进的LLM仍然容易受到越狱的影响，这些越狱巧妙地重新表达有害提示，同时也过度拒绝安全但措辞合理的查询。最后，我们提出了一种缓解策略，使用轻量级分类器来检测越狱尝试，并通过基于条件反射来引导模型转向更安全的行为。CARES提供了一个严格的框架，用于在对抗和模糊的条件下测试和改善医学LLM安全性。



## **40. Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities**

基于有限样本浓度不等式的LLM文本检测零次统计检验 stat.ML

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2501.02406v4) [paper-pdf](http://arxiv.org/pdf/2501.02406v4)

**Authors**: Tara Radvand, Mojtaba Abdolmaleki, Mohamed Mostagir, Ambuj Tewari

**Abstract**: Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly challenging as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. In this paper, we answer the following question: Given a piece of text, can we identify whether it was produced by a particular LLM or not? We model LLM-generated text as a sequential stochastic process with complete dependence on history. We then design zero-shot statistical tests to (i) distinguish between text generated by two different known sets of LLMs $A$ (non-sanctioned) and $B$ (in-house), and (ii) identify whether text was generated by a known LLM or generated by any unknown model, e.g., a human or some other language generation process. We prove that the type I and type II errors of our test decrease exponentially with the length of the text. For that, we show that if $B$ generates the text, then except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. We then present experiments using LLMs with white-box access to support our theoretical results and empirically examine the robustness of our results to black-box settings and adversarial attacks. In the black-box setting, our method achieves an average TPR of 82.5\% at a fixed FPR of 5\%. Under adversarial perturbations, our minimum TPR is 48.6\% at the same FPR threshold. Both results outperform all non-commercial baselines. See https://github.com/TaraRadvand74/llm-text-detection for code, data, and an online demo of the project.

摘要: 验证内容的出处对于许多组织的功能至关重要，例如，教育机构、社交媒体平台、公司等。随着大型语言模型（LLM）生成的文本与人类生成的内容几乎无法区分，这个问题变得越来越具有挑战性。此外，许多机构利用内部LLM，并希望确保外部未经批准的LLM不会在机构内制作内容。在本文中，我们回答了以下问题：给定一段文本，我们能否识别它是否是由特定的LLM生成的？我们将LLM生成的文本建模为一个完全依赖于历史的顺序随机过程。然后，我们设计零镜头统计测试，以（i）区分由两组不同的已知LLM $A$（未经批准）和$B$（内部）生成的文本，以及（ii）识别文本是由已知LLM生成还是由任何未知模型生成，例如人类或某种其他语言生成过程。我们证明，我们测试的I型和II型错误随着文本长度的增加而呈指数级减少。为此，我们表明，如果$B$生成文本，那么除了字符串长度的概率呈指数级小外，$A$下的字符串的log困惑度会收敛到$B$和$A$的平均交叉熵。然后，我们使用具有白盒访问权限的LLM进行了实验，以支持我们的理论结果，并从经验上检查我们的结果对黑匣子设置和对抗攻击的鲁棒性。在黑匣子设置中，我们的方法在固定FPR为5%时实现了82.5%的平均TPA。在对抗性扰动下，在相同FPR阈值下，我们的最低TPA为48.6%。这两个结果都优于所有非商业基线。请参阅https://github.com/TaraRadvand74/llm-text-detection了解该项目的代码、数据和在线演示。



## **41. MPMA: Preference Manipulation Attack Against Model Context Protocol**

MPMA：针对模型上下文协议的偏好操纵攻击 cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11154v1) [paper-pdf](http://arxiv.org/pdf/2505.11154v1)

**Authors**: Zihan Wang, Hongwei Li, Rui Zhang, Yu Liu, Wenbo Jiang, Wenshu Fan, Qingchuan Zhao, Guowen Xu

**Abstract**: Model Context Protocol (MCP) standardizes interface mapping for large language models (LLMs) to access external data and tools, which revolutionizes the paradigm of tool selection and facilitates the rapid expansion of the LLM agent tool ecosystem. However, as the MCP is increasingly adopted, third-party customized versions of the MCP server expose potential security vulnerabilities. In this paper, we first introduce a novel security threat, which we term the MCP Preference Manipulation Attack (MPMA). An attacker deploys a customized MCP server to manipulate LLMs, causing them to prioritize it over other competing MCP servers. This can result in economic benefits for attackers, such as revenue from paid MCP services or advertising income generated from free servers. To achieve MPMA, we first design a Direct Preference Manipulation Attack ($\mathtt{DPMA}$) that achieves significant effectiveness by inserting the manipulative word and phrases into the tool name and description. However, such a direct modification is obvious to users and lacks stealthiness. To address these limitations, we further propose Genetic-based Advertising Preference Manipulation Attack ($\mathtt{GAPMA}$). $\mathtt{GAPMA}$ employs four commonly used strategies to initialize descriptions and integrates a Genetic Algorithm (GA) to enhance stealthiness. The experiment results demonstrate that $\mathtt{GAPMA}$ balances high effectiveness and stealthiness. Our study reveals a critical vulnerability of the MCP in open ecosystems, highlighting an urgent need for robust defense mechanisms to ensure the fairness of the MCP ecosystem.

摘要: 模型上下文协议（HCP）将大型语言模型（LLM）的接口映射同步化，以访问外部数据和工具，这彻底改变了工具选择的范式，并促进了LLM代理工具生态系统的快速扩展。然而，随着LCP越来越多地采用，第三方定制版本的LCP服务器暴露了潜在的安全漏洞。在本文中，我们首先介绍了一种新型的安全威胁，我们将其称为LCP偏好操纵攻击（MPMA）。攻击者部署自定义的LCP服务器来操纵LLM，导致他们将其优先于其他竞争的LCP服务器。这可能会为攻击者带来经济利益，例如付费HCP服务的收入或免费服务器产生的广告收入。为了实现MPMA，我们首先设计了直接偏好操纵攻击（$\mathtt {DPMA}$），该攻击通过将操纵性单词和短语插入工具名称和描述中来实现显着的有效性。但这样的直接修改对于用户来说是显而易见的，缺乏隐蔽性。为了解决这些限制，我们进一步提出了基于遗传的广告偏好操纵攻击（$\mathtt {GAPMA}$）。$\mathtt {GAPMA}$采用四种常用策略来初始化描述，并集成遗传算法（GA）来增强隐蔽性。实验结果表明$\mathtt {GAPMA}$平衡了高效率和隐蔽性。我们的研究揭示了开放生态系统中的重要脆弱性，凸显了迫切需要强大的防御机制来确保CP生态系统的公平性。



## **42. SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models**

SCAM：多模式基础模型的真实印刷稳健性评估 cs.CV

Accepted at CVPR 2025 Workshop EVAL-FoMo-2

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2504.04893v4) [paper-pdf](http://arxiv.org/pdf/2504.04893v4)

**Authors**: Justus Westerhoff, Erblina Purelku, Jakob Hackstein, Jonas Loos, Leo Pinetzki, Lorenz Hufe

**Abstract**: Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper along with the code for evaluations at www.bliss.berlin/research/scam.

摘要: 排版攻击利用多模态基础模型中文本和视觉内容之间的相互作用，当误导性文本嵌入图像中时会导致错误分类。然而，现有的数据集在规模和多样性方面有限，因此难以研究这些脆弱性。在本文中，我们介绍了SCAM，这是迄今为止最大、最多样化的现实世界印刷攻击图像数据集，包含涵盖数百个对象类别和攻击词的1，162张图像。通过对SCAM上的视觉语言模型（VLM）进行广泛的基准测试，我们证明了排版攻击会显着降低性能，并确定训练数据和模型架构会影响对这些攻击的敏感性。我们的研究结果表明，由于视觉编码器的选择，印刷攻击在最先进的大型视觉语言模型（LVLM）中持续存在，尽管更大的大型语言模型（LLM）主干有助于减轻它们的脆弱性。此外，我们还证明合成攻击与现实世界（手写）攻击非常相似，验证了它们在研究中的用途。我们的工作提供了全面的资源和经验见解，以促进未来对稳健且值得信赖的多模式人工智能系统的研究。我们在www.bliss.berlin/research/scam上公开发布本文中介绍的数据集以及评估代码。



## **43. PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization**

PIG：通过基于对象的迭代上下文优化对LLM进行隐私越狱攻击 cs.CR

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.09921v2) [paper-pdf](http://arxiv.org/pdf/2505.09921v2)

**Authors**: Yidan Wang, Yanan Cao, Yubing Ren, Fang Fang, Zheng Lin, Binxing Fang

**Abstract**: Large Language Models (LLMs) excel in various domains but pose inherent privacy risks. Existing methods to evaluate privacy leakage in LLMs often use memorized prefixes or simple instructions to extract data, both of which well-alignment models can easily block. Meanwhile, Jailbreak attacks bypass LLM safety mechanisms to generate harmful content, but their role in privacy scenarios remains underexplored. In this paper, we examine the effectiveness of jailbreak attacks in extracting sensitive information, bridging privacy leakage and jailbreak attacks in LLMs. Moreover, we propose PIG, a novel framework targeting Personally Identifiable Information (PII) and addressing the limitations of current jailbreak methods. Specifically, PIG identifies PII entities and their types in privacy queries, uses in-context learning to build a privacy context, and iteratively updates it with three gradient-based strategies to elicit target PII. We evaluate PIG and existing jailbreak methods using two privacy-related datasets. Experiments on four white-box and two black-box LLMs show that PIG outperforms baseline methods and achieves state-of-the-art (SoTA) results. The results underscore significant privacy risks in LLMs, emphasizing the need for stronger safeguards. Our code is availble at https://github.com/redwyd/PrivacyJailbreak.

摘要: 大型语言模型（LLM）在各个领域都表现出色，但也存在固有的隐私风险。评估LLM隐私泄露的现有方法通常使用记忆的前置码或简单指令来提取数据，而良好对齐的模型可以轻松阻止这两种情况。与此同时，越狱攻击绕过了LLM安全机制来生成有害内容，但它们在隐私场景中的作用仍然没有得到充分研究。在本文中，我们研究了越狱攻击在提取敏感信息、弥合LLC中隐私泄露和越狱攻击方面的有效性。此外，我们还提出了PIG，这是一种针对个人可识别信息（PRI）并解决当前越狱方法的局限性的新型框架。具体来说，PIG识别隐私查询中的PRI实体及其类型，使用上下文学习来构建隐私上下文，并使用三种基于梯度的策略迭代更新它以引出目标PRI。我们使用两个与隐私相关的数据集评估PIG和现有的越狱方法。对四个白盒和两个黑盒LLM的实验表明，PIG优于基线方法并实现了最先进的（SoTA）结果。结果强调了LLM中存在的重大隐私风险，强调了加强保护措施的必要性。我们的代码可在https://github.com/redwyd/PrivacyJailbreak上获取。



## **44. ACSE-Eval: Can LLMs threat model real-world cloud infrastructure?**

ACSE-Eval：LLM威胁能否建模现实世界的云基础设施？ cs.CR

Submitted to the 39th Annual Conference on Neural Information  Processing Systems

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11565v1) [paper-pdf](http://arxiv.org/pdf/2505.11565v1)

**Authors**: Sarthak Munshi, Swapnil Pathak, Sonam Ghatode, Thenuga Priyadarshini, Dhivya Chandramouleeswaran, Ashutosh Rana

**Abstract**: While Large Language Models have shown promise in cybersecurity applications, their effectiveness in identifying security threats within cloud deployments remains unexplored. This paper introduces AWS Cloud Security Engineering Eval, a novel dataset for evaluating LLMs cloud security threat modeling capabilities. ACSE-Eval contains 100 production grade AWS deployment scenarios, each featuring detailed architectural specifications, Infrastructure as Code implementations, documented security vulnerabilities, and associated threat modeling parameters. Our dataset enables systemic assessment of LLMs abilities to identify security risks, analyze attack vectors, and propose mitigation strategies in cloud environments. Our evaluations on ACSE-Eval demonstrate that GPT 4.1 and Gemini 2.5 Pro excel at threat identification, with Gemini 2.5 Pro performing optimally in 0-shot scenarios and GPT 4.1 showing superior results in few-shot settings. While GPT 4.1 maintains a slight overall performance advantage, Claude 3.7 Sonnet generates the most semantically sophisticated threat models but struggles with threat categorization and generalization. To promote reproducibility and advance research in automated cybersecurity threat analysis, we open-source our dataset, evaluation metrics, and methodologies.

摘要: 虽然大型语言模型在网络安全应用中表现出了希望，但它们在识别云部署中安全威胁方面的有效性仍有待探索。本文介绍了AWS云安全工程Eval，这是一个用于评估LLM云安全威胁建模能力的新型数据集。ACSE-Eval包含100个生产级AWS部署场景，每个场景都包含详细的架构规范、基础设施即代码实现、记录在案的安全漏洞和相关的威胁建模参数。我们的数据集能够系统评估LLM识别安全风险、分析攻击载体并在云环境中提出缓解策略的能力。我们对ACSE-Eval的评估表明，GPT 4.1和Gemini 2.5 Pro在威胁识别方面表现出色，Gemini 2.5 Pro在0次射击场景中表现最佳，GPT 4.1在几次射击设置中表现出色。虽然GPT 4.1保持了轻微的整体性能优势，但Claude 3.7 Sonnet生成了语义上最复杂的威胁模型，但在威胁分类和泛化方面遇到了困难。为了促进自动化网络安全威胁分析的可重复性和推进研究，我们开源了我们的数据集，评估指标和方法。



## **45. LARGO: Latent Adversarial Reflection through Gradient Optimization for Jailbreaking LLMs**

LARGO：通过越狱LLM的梯度优化实现潜在的对抗反射 cs.LG

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.10838v1) [paper-pdf](http://arxiv.org/pdf/2505.10838v1)

**Authors**: Ran Li, Hao Wang, Chengzhi Mao

**Abstract**: Efficient red-teaming method to uncover vulnerabilities in Large Language Models (LLMs) is crucial. While recent attacks often use LLMs as optimizers, the discrete language space make gradient-based methods struggle. We introduce LARGO (Latent Adversarial Reflection through Gradient Optimization), a novel latent self-reflection attack that reasserts the power of gradient-based optimization for generating fluent jailbreaking prompts. By operating within the LLM's continuous latent space, LARGO first optimizes an adversarial latent vector and then recursively call the same LLM to decode the latent into natural language. This methodology yields a fast, effective, and transferable attack that produces fluent and stealthy prompts. On standard benchmarks like AdvBench and JailbreakBench, LARGO surpasses leading jailbreaking techniques, including AutoDAN, by 44 points in attack success rate. Our findings demonstrate a potent alternative to agentic LLM prompting, highlighting the efficacy of interpreting and attacking LLM internals through gradient optimization.

摘要: 发现大型语言模型（LLM）中漏洞的高效红色团队方法至关重要。虽然最近的攻击经常使用LLM作为优化器，但离散语言空间使得基于梯度的方法变得困难。我们引入了LARGO（通过梯度优化的潜在对抗反射），这是一种新型的潜在自我反射攻击，它重申了基于梯度的优化用于生成流畅的越狱提示的力量。通过在LLM的连续潜在空间内操作，LARGO首先优化对抗性潜在载体，然后循环调用相同的LLM将潜在载体解码为自然语言。这种方法可以产生快速、有效且可转移的攻击，从而产生流畅且隐蔽的提示。在AdvBench和JailbreakBench等标准基准上，LARGO的攻击成功率比AutoDAN等领先越狱技术高出44分。我们的研究结果证明了代理LLM提示的有效替代方案，强调了通过梯度优化解释和攻击LLM内部内容的有效性。



## **46. SecReEvalBench: A Multi-turned Security Resilience Evaluation Benchmark for Large Language Models**

SecReEvalBench：大型语言模型的多角度安全弹性评估基准 cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.07584v2) [paper-pdf](http://arxiv.org/pdf/2505.07584v2)

**Authors**: Huining Cui, Wei Liu

**Abstract**: The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cybersecurity attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turn attacks. To address this gap, we present SecReEvalBench, the Security Resilience Evaluation Benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain-Based Attack Resilience Score and Chain-Based Attack Rejection Time Score. Moreover, SecReEvalBench employs six questioning sequences for model assessment: one-off attack, successive attack, successive reverse attack, alternative attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels. In addition, we introduce a dataset customized for the benchmark, which incorporates both neutral and malicious prompts, categorised across seven security domains and sixteen attack techniques. In applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats. The SecReEvalBench dataset is publicly available at https://kaggle.com/datasets/5a7ee22cf9dab6c93b55a73f630f6c9b42e936351b0ae98fbae6ddaca7fe248d, which provides a groundwork for advancing research in large language model security.

摘要: 大型语言模型在安全敏感领域的部署越来越多，需要严格评估它们对抗基于预算的敌对攻击的弹性。虽然之前的基准侧重于有限且预定义的攻击域（例如网络安全攻击）的安全评估，但它们通常缺乏对意图驱动的对抗提示的全面评估以及对现实生活中基于情景的多回合攻击的考虑。为了解决这一差距，我们提出了SecReEvalBench，安全韧性评估基准，它定义了四个新颖的指标：即时攻击韧性分数、即时攻击拒绝逻辑分数、基于链的攻击韧性分数和基于链的攻击拒绝时间分数。此外，SecReEvalBench采用六个提问序列进行模型评估：一次性攻击、连续攻击、连续反向攻击、替代攻击、威胁级别不断上升的顺序上升攻击和威胁级别不断下降的顺序下降攻击。此外，我们还引入了一个为基准定制的数据集，其中包含中性和恶意提示，分为七个安全域和十六种攻击技术。在应用该基准时，我们系统地评估了五个最先进的开放加权大型语言模型：Llama 3.1、Gemma 2、Mistral v0.3、DeepSeek-R1和Qwen 3。我们的研究结果为现代大型语言模型在防御不断变化的对抗威胁方面的优势和弱点提供了重要的见解。SecReEvalBench数据集可在https：//kaggle.com/guardets/5a7ee22CF9dab6c93b55a73f630f6c9 b42 e936351 b 0ae 98 fbae 6ddaca 7 fe 248 d上公开，为推进大型语言模型安全性研究提供了基础。



## **47. Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines**

大型语言模型搜索引擎的对抗性攻击动态 cs.CL

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2501.00745v2) [paper-pdf](http://arxiv.org/pdf/2501.00745v2)

**Authors**: Xiyang Hu

**Abstract**: The increasing integration of Large Language Model (LLM) based search engines has transformed the landscape of information retrieval. However, these systems are vulnerable to adversarial attacks, especially ranking manipulation attacks, where attackers craft webpage content to manipulate the LLM's ranking and promote specific content, gaining an unfair advantage over competitors. In this paper, we study the dynamics of ranking manipulation attacks. We frame this problem as an Infinitely Repeated Prisoners' Dilemma, where multiple players strategically decide whether to cooperate or attack. We analyze the conditions under which cooperation can be sustained, identifying key factors such as attack costs, discount rates, attack success rates, and trigger strategies that influence player behavior. We identify tipping points in the system dynamics, demonstrating that cooperation is more likely to be sustained when players are forward-looking. However, from a defense perspective, we find that simply reducing attack success probabilities can, paradoxically, incentivize attacks under certain conditions. Furthermore, defensive measures to cap the upper bound of attack success rates may prove futile in some scenarios. These insights highlight the complexity of securing LLM-based systems. Our work provides a theoretical foundation and practical insights for understanding and mitigating their vulnerabilities, while emphasizing the importance of adaptive security strategies and thoughtful ecosystem design.

摘要: 基于大语言模型（LLM）的搜索引擎的日益集成已经改变了信息检索的格局。然而，这些系统很容易受到对抗攻击，尤其是排名操纵攻击，攻击者精心制作网页内容来操纵LLM的排名并推广特定内容，从而获得相对于竞争对手的不公平优势。本文中，我们研究了排名操纵攻击的动力学。我们将这个问题定义为“无限重复的囚徒困境”，其中多个参与者战略性地决定是合作还是攻击。我们分析了持续合作的条件，确定了攻击成本、折扣率、攻击成功率以及影响玩家行为的触发策略等关键因素。我们确定了系统动态中的临界点，证明当参与者具有前瞻性时，合作更有可能持续。然而，从防御的角度来看，我们发现，矛盾的是，简单地降低攻击成功概率就可以在某些条件下激励攻击。此外，在某些情况下，限制攻击成功率上限的防御措施可能是徒劳的。这些见解突出了保护基于LLM的系统的复杂性。我们的工作为理解和减轻其脆弱性提供了理论基础和实践见解，同时强调了自适应安全策略和周到的生态系统设计的重要性。



## **48. S3C2 Summit 2024-09: Industry Secure Software Supply Chain Summit**

S3 C2峰会2024-09：行业安全软件供应链峰会 cs.CR

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.10538v1) [paper-pdf](http://arxiv.org/pdf/2505.10538v1)

**Authors**: Imranur Rahman, Yasemin Acar, Michel Cukier, William Enck, Christian Kastner, Alexandros Kapravelos, Dominik Wermke, Laurie Williams

**Abstract**: While providing economic and software development value, software supply chains are only as strong as their weakest link. Over the past several years, there has been an exponential increase in cyberattacks, specifically targeting vulnerable links in critical software supply chains. These attacks disrupt the day-to-day functioning and threaten the security of nearly everyone on the internet, from billion-dollar companies and government agencies to hobbyist open-source developers. The ever-evolving threat of software supply chain attacks has garnered interest from the software industry and the US government in improving software supply chain security.   On September 20, 2024, three researchers from the NSF-backed Secure Software Supply Chain Center (S3C2) conducted a Secure Software Supply Chain Summit with a diverse set of 12 practitioners from 9 companies. The goals of the Summit were to: (1) to enable sharing between individuals from different companies regarding practical experiences and challenges with software supply chain security, (2) to help form new collaborations, (3) to share our observations from our previous summits with industry, and (4) to learn about practitioners' challenges to inform our future research direction. The summit consisted of discussions of six topics relevant to the companies represented, including updating vulnerable dependencies, component and container choice, malicious commits, building infrastructure, large language models, and reducing entire classes of vulnerabilities.

摘要: 在提供经济和软件开发价值的同时，软件供应链的强大程度取决于其最薄弱的环节。在过去的几年里，网络攻击呈指数级增加，特别是针对关键软件供应链中的脆弱环节。这些攻击扰乱了日常运作，并威胁到互联网上几乎所有人的安全，从价值数十亿美元的公司和政府机构到爱好者的开源开发人员。软件供应链攻击的不断变化的威胁引起了软件行业和美国政府对改善软件供应链安全的兴趣。   2024年9月20日，来自NSF支持的安全软件供应链中心（S3 C2）的三名研究人员与来自9家公司的12名从业者举行了安全软件供应链峰会。峰会的目标是：（1）实现来自不同公司的个人之间就软件供应链安全方面的实践经验和挑战进行分享，（2）帮助形成新的合作，（3）与行业分享我们在之前峰会上的观察，（4）了解从业者面临的挑战，为我们未来的研究方向提供信息。峰会讨论了与与会公司相关的六个主题，包括更新脆弱依赖关系、组件和容器选择、恶意提交、构建基础设施、大型语言模型以及减少整个漏洞类别。



## **49. MapExplorer: New Content Generation from Low-Dimensional Visualizations**

MapExplorer：来自低维可视化的新内容生成 cs.AI

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2412.18673v2) [paper-pdf](http://arxiv.org/pdf/2412.18673v2)

**Authors**: Xingjian Zhang, Ziyang Xiong, Shixuan Liu, Yutong Xie, Tolga Ergen, Dongsub Shim, Hua Xu, Honglak Lee, Qiaozhu Me

**Abstract**: Low-dimensional visualizations, or "projection maps," are widely used in scientific and creative domains to interpret large-scale and complex datasets. These visualizations not only aid in understanding existing knowledge spaces but also implicitly guide exploration into unknown areas. Although techniques such as t-SNE and UMAP can generate these maps, there exists no systematic method for leveraging them to generate new content. To address this, we introduce MapExplorer, a novel knowledge discovery task that translates coordinates within any projection map into coherent, contextually aligned textual content. This allows users to interactively explore and uncover insights embedded in the maps. To evaluate the performance of MapExplorer methods, we propose Atometric, a fine-grained metric inspired by ROUGE that quantifies logical coherence and alignment between generated and reference text. Experiments on diverse datasets demonstrate the versatility of MapExplorer in generating scientific hypotheses, crafting synthetic personas, and devising strategies for attacking large language models-even with simple baseline methods. By bridging visualization and generation, our work highlights the potential of MapExplorer to enable intuitive human-AI collaboration in large-scale data exploration.

摘要: 低维可视化或“投影地图”广泛用于科学和创意领域，以解释大规模和复杂的数据集。这些可视化不仅有助于理解现有的知识空间，而且还隐含地指导对未知领域的探索。尽管t-SNE和UMAP等技术可以生成这些地图，但不存在利用它们来生成新内容的系统方法。为了解决这个问题，我们引入了MapExplorer，这是一项新颖的知识发现任务，可以将任何投影地图内的坐标转换为连贯、上下文对齐的文本内容。这允许用户交互式探索和发现地图中嵌入的见解。为了评估MapExplorer方法的性能，我们提出了Atric，这是一种受ROUGE启发的细粒度指标，可以量化生成文本和参考文本之间的逻辑一致性和一致性。对不同数据集的实验证明了MapExplorer在生成科学假设、制作合成人物角色以及设计攻击大型语言模型的策略方面的多功能性--即使使用简单的基线方法。通过连接可视化和生成，我们的工作凸显了MapExplorer在大规模数据探索中实现直观的人机协作的潜力。



## **50. Scaling Laws for Black box Adversarial Attacks**

黑匣子对抗攻击的缩放定律 cs.LG

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2411.16782v2) [paper-pdf](http://arxiv.org/pdf/2411.16782v2)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.

摘要: 对抗性示例通常表现出良好的跨模型可移植性，从而能够在有关其架构和参数的有限信息的情况下对黑匣子模型进行攻击，这在商业黑匣子场景中具有高度威胁性。模型集成是通过攻击多个代理模型来提高对抗性示例可移植性的有效策略。然而，由于之前的研究通常在整体中采用很少的模型，因此扩大模型数量是否可以进一步改善黑匣子攻击仍然是一个悬而未决的问题。受大型基金会模型缩放定律的启发，我们在这项工作中研究了黑匣子对抗攻击的缩放定律。通过理论分析和实证评估，我们得出了明确的缩放定律，即使用更多的代理模型增强了对抗性可转让性。全面的实验验证了标准图像分类器、多样化防御模型和使用各种对抗攻击方法的多模式大型语言模型的主张。具体来说，通过缩放定律，即使是GPT-4 o等专有模型，我们也能实现90%以上的传输攻击成功率。进一步的可视化表明，对抗性扰动的可解释性和语义也存在缩放定律。



