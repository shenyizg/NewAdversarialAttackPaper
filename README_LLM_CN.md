# Latest Large Language Model Attack Papers
**update at 2025-09-19 15:27:53**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Evil Vizier: Vulnerabilities of LLM-Integrated XR Systems**

Evil Vizier：LLM集成XR系统的漏洞 cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15213v1) [paper-pdf](http://arxiv.org/pdf/2509.15213v1)

**Authors**: Yicheng Zhang, Zijian Huang, Sophie Chen, Erfan Shayegani, Jiasi Chen, Nael Abu-Ghazaleh

**Abstract**: Extended reality (XR) applications increasingly integrate Large Language Models (LLMs) to enhance user experience, scene understanding, and even generate executable XR content, and are often called "AI glasses". Despite these potential benefits, the integrated XR-LLM pipeline makes XR applications vulnerable to new forms of attacks. In this paper, we analyze LLM-Integated XR systems in the literature and in practice and categorize them along different dimensions from a systems perspective. Building on this categorization, we identify a common threat model and demonstrate a series of proof-of-concept attacks on multiple XR platforms that employ various LLM models (Meta Quest 3, Meta Ray-Ban, Android, and Microsoft HoloLens 2 running Llama and GPT models). Although these platforms each implement LLM integration differently, they share vulnerabilities where an attacker can modify the public context surrounding a legitimate LLM query, resulting in erroneous visual or auditory feedback to users, thus compromising their safety or privacy, sowing confusion, or other harmful effects. To defend against these threats, we discuss mitigation strategies and best practices for developers, including an initial defense prototype, and call on the community to develop new protection mechanisms to mitigate these risks.

摘要: 延展实境（XR）应用程序越来越多地集成大型语言模型（LLM），以增强用户体验、场景理解，甚至生成可执行XR内容，通常被称为“AI眼镜”。尽管有这些潜在的好处，但集成的XR-LLM管道使XR应用程序容易受到新形式的攻击。在本文中，我们分析了LLM-Integated XR系统在文献和实践中，并从系统的角度沿着不同的维度对它们进行分类。在此分类的基础上，我们识别了常见的威胁模型，并在采用各种LLM模型（Meta Quest 3、Meta Ray-Ban、Android和运行Lama和GPT模型的Microsoft HoloLens 2）的多个XR平台上演示了一系列概念验证攻击。尽管这些平台各自以不同的方式实现LLM集成，但它们都有漏洞，攻击者可以修改围绕合法LLM查询的公共上下文，从而导致向用户提供错误的视觉或听觉反馈，从而损害他们的安全或隐私、散布混乱或其他有害影响。为了抵御这些威胁，我们讨论了开发人员的缓解策略和最佳实践，包括初始的防御原型，并呼吁社区开发新的保护机制来缓解这些风险。



## **2. Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction**

超越表面对齐：通过概率简化拒绝指示重建LLM安全机制 cs.CR

Accepted by EMNLP2025 Finding

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15202v1) [paper-pdf](http://arxiv.org/pdf/2509.15202v1)

**Authors**: Yuanbo Xie, Yingjie Zhang, Tianyun Liu, Duohe Ma, Tingwen Liu

**Abstract**: Jailbreak attacks pose persistent threats to large language models (LLMs). Current safety alignment methods have attempted to address these issues, but they experience two significant limitations: insufficient safety alignment depth and unrobust internal defense mechanisms. These limitations make them vulnerable to adversarial attacks such as prefilling and refusal direction manipulation. We introduce DeepRefusal, a robust safety alignment framework that overcomes these issues. DeepRefusal forces the model to dynamically rebuild its refusal mechanisms from jailbreak states. This is achieved by probabilistically ablating the refusal direction across layers and token depths during fine-tuning. Our method not only defends against prefilling and refusal direction attacks but also demonstrates strong resilience against other unseen jailbreak strategies. Extensive evaluations on four open-source LLM families and six representative attacks show that DeepRefusal reduces attack success rates by approximately 95%, while maintaining model capabilities with minimal performance degradation.

摘要: 越狱攻击对大型语言模型（LLM）构成持续威胁。当前的安全对齐方法试图解决这些问题，但它们遇到了两个重大局限性：安全对齐深度不足和内部防御机制不健全。这些限制使它们容易受到预填充和拒绝方向操纵等敌对攻击。我们引入DeepRefusal，这是一个强大的安全调整框架，可以克服这些问题。DeepRefusal迫使该模型动态重建其来自越狱国家的拒绝机制。这是通过在微调期间概率消除跨层和代币深度的拒绝方向来实现的。我们的方法不仅可以抵御预填充和拒绝方向攻击，而且还表现出对其他看不见的越狱策略的强大韧性。对四个开源LLM系列和六种代表性攻击的广泛评估表明，DeepRefusal将攻击成功率降低了约95%，同时以最小的性能下降保持模型能力。



## **3. Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System**

在基于LLM的开发系统中利用工具调用提示实现工具行为劫持 cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.05755v3) [paper-pdf](http://arxiv.org/pdf/2509.05755v3)

**Authors**: Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Yu Liu, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.

摘要: 基于法学硕士的代理系统利用大型语言模型来处理用户查询、做出决策并执行外部工具，以执行跨聊天机器人、客户服务和软件工程等领域的复杂任务。这些系统的一个关键组件是工具调用提示（TIP），它定义了工具交互协议并指导LLM确保工具使用的安全性和正确性。尽管TIP的安全性很重要，但在很大程度上被忽视了。这项工作调查了与TIP相关的安全风险，揭示了Cursor、Claude Code等基于LLM的主要系统容易受到远程代码执行（RCE）和拒绝服务（NOS）等攻击。通过系统性TIP利用工作流程（TEW），我们通过操纵工具调用演示了外部工具行为劫持。我们还提出了防御机制来增强基于LLM的代理系统中的TIP安全性。



## **4. QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA**

QA-LIGN：通过宪法分解的QA调整LLM cs.CL

Accepted to Findings of EMNLP 2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2506.08123v3) [paper-pdf](http://arxiv.org/pdf/2506.08123v3)

**Authors**: Jacob Dineen, Aswin RRV, Qin Liu, Zhikun Xu, Xiao Ye, Ming Shen, Zhaonan Li, Shijie Lu, Chitta Baral, Muhao Chen, Ben Zhou

**Abstract**: Alignment of large language models (LLMs) with principles like helpfulness, honesty, and harmlessness typically relies on scalar rewards that obscure which objectives drive the training signal. We introduce QA-LIGN, which decomposes monolithic rewards into interpretable principle-specific evaluations through structured natural language programs. Models learn through a draft, critique, and revise pipeline, where symbolic evaluation against the rubrics provides transparent feedback for both initial and revised responses during GRPO training. Applied to uncensored Llama-3.1-8B-Instruct, QA-LIGN reduces attack success rates by up to 68.7% while maintaining a 0.67% false refusal rate, achieving Pareto optimal safety-helpfulness performance and outperforming both DPO and GRPO with state-of-the-art reward models given equivalent training. These results demonstrate that making reward signals interpretable and modular improves alignment effectiveness, suggesting transparency enhances LLM safety.

摘要: 大型语言模型（LLM）与乐于助人、诚实和无害等原则的一致通常依赖于量化奖励，这些奖励模糊了哪些目标驱动训练信号。我们引入QA-LIGN，它通过结构化自然语言程序将单一奖励分解为可解释的特定于原则的评估。模型通过起草、评论和修改管道进行学习，其中针对主题的象征性评估为GRPO培训期间的初始和修改响应提供透明的反馈。应用于未经审查的Llama-3.1- 8B-Direct，QA-LIGN将攻击成功率降低高达68.7%，同时保持0.67%的错误拒绝率，实现了帕累托最佳安全帮助性能，并在同等培训的情况下优于DPO和GRPO。这些结果表明，使奖励信号可解释和模块化可以提高对齐有效性，这表明透明度增强了LLM的安全性。



## **5. AIP: Subverting Retrieval-Augmented Generation via Adversarial Instructional Prompt**

AIP：通过对抗性指令提示颠覆检索增强生成 cs.CV

Accepted at EMNLP 2025 Conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15159v1) [paper-pdf](http://arxiv.org/pdf/2509.15159v1)

**Authors**: Saket S. Chaturvedi, Gaurav Bagwe, Lan Zhang, Xiaoyong Yuan

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by retrieving relevant documents from external sources to improve factual accuracy and verifiability. However, this reliance introduces new attack surfaces within the retrieval pipeline, beyond the LLM itself. While prior RAG attacks have exposed such vulnerabilities, they largely rely on manipulating user queries, which is often infeasible in practice due to fixed or protected user inputs. This narrow focus overlooks a more realistic and stealthy vector: instructional prompts, which are widely reused, publicly shared, and rarely audited. Their implicit trust makes them a compelling target for adversaries to manipulate RAG behavior covertly.   We introduce a novel attack for Adversarial Instructional Prompt (AIP) that exploits adversarial instructional prompts to manipulate RAG outputs by subtly altering retrieval behavior. By shifting the attack surface to the instructional prompts, AIP reveals how trusted yet seemingly benign interface components can be weaponized to degrade system integrity. The attack is crafted to achieve three goals: (1) naturalness, to evade user detection; (2) utility, to encourage use of prompts; and (3) robustness, to remain effective across diverse query variations. We propose a diverse query generation strategy that simulates realistic linguistic variation in user queries, enabling the discovery of prompts that generalize across paraphrases and rephrasings. Building on this, a genetic algorithm-based joint optimization is developed to evolve adversarial prompts by balancing attack success, clean-task utility, and stealthiness. Experimental results show that AIP achieves up to 95.23% ASR while preserving benign functionality. These findings uncover a critical and previously overlooked vulnerability in RAG systems, emphasizing the need to reassess the shared instructional prompts.

摘要: 检索增强生成（RAG）通过从外部源检索相关文档来增强大型语言模型（LLM），以提高事实准确性和可验证性。然而，这种依赖在LLM本身之外的检索管道中引入了新的攻击表面。虽然之前的RAG攻击已经暴露了此类漏洞，但它们在很大程度上依赖于操纵用户查询，而由于用户输入固定或受保护，这在实践中通常是不可行的。这种狭隘的焦点忽视了一个更现实、更隐蔽的载体：教学提示，它们被广泛重复使用、公开共享，而且很少审计。他们的隐性信任使他们成为对手秘密操纵RAG行为的引人注目的目标。   我们引入了一种针对对抗性教学提示（AIP）的新型攻击，该攻击利用对抗性教学提示通过微妙地改变检索行为来操纵RAG输出。通过将攻击面转移到指令提示，AIP揭示了如何将可信但看似良性的接口组件武器化以降低系统完整性。该攻击旨在实现三个目标：（1）自然性，以逃避用户检测;（2）实用性，以鼓励使用提示;（3）稳健性，以在不同的查询变体中保持有效。我们提出了一种多样化的查询生成策略，该策略模拟用户查询中现实的语言变化，从而能够发现在重述和改写中进行概括的提示。在此基础上，开发了基于遗传算法的联合优化，通过平衡攻击成功、干净任务效用和隐蔽性来进化对抗提示。实验结果表明，AIP在保持良性功能的同时实现了高达95.23%的ASB。这些发现揭示了RAG系统中一个以前被忽视的关键漏洞，强调需要重新评估共享的教学提示。



## **6. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2409.13174v3) [paper-pdf](http://arxiv.org/pdf/2409.13174v3)

**Authors**: Hao Cheng, Erjia Xiao, Yichi Wang, Chengyuan Yu, Mengshu Sun, Qiang Zhang, Yijie Guo, Kaidi Xu, Jize Zhang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.

摘要: 最近，在多模式大型语言模型（MLLM）进步的推动下，人们提出了视觉语言动作模型（VLAM），以在机器人操纵任务的开放词汇场景中实现更好的性能。由于操纵任务涉及与物理世界的直接互动，因此在执行该任务期间确保稳健性和安全性始终是一个非常关键的问题。本文通过综合当前对MLLM的安全性研究以及物理世界中操纵任务的具体应用场景，对VLAM在潜在物理威胁面前进行了全面评估。具体来说，我们提出了物理脆弱性评估管道（PVEP），它可以整合尽可能多的视觉模式物理威胁，以评估VLAM的物理稳健性。PVEP中的物理威胁具体包括分发外、基于印刷术的视觉提示和对抗性补丁攻击。通过比较VLAM受到攻击前后的性能波动，我们提供了VLAM如何响应不同物理威胁的可概括的\textBF{\textit{Analyses}。



## **7. Sentinel Agents for Secure and Trustworthy Agentic AI in Multi-Agent Systems**

在多代理系统中实现安全且值得信赖的大型人工智能的哨兵代理 cs.AI

25 pages, 12 figures

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14956v1) [paper-pdf](http://arxiv.org/pdf/2509.14956v1)

**Authors**: Diego Gosmar, Deborah A. Dahl

**Abstract**: This paper proposes a novel architectural framework aimed at enhancing security and reliability in multi-agent systems (MAS). A central component of this framework is a network of Sentinel Agents, functioning as a distributed security layer that integrates techniques such as semantic analysis via large language models (LLMs), behavioral analytics, retrieval-augmented verification, and cross-agent anomaly detection. Such agents can potentially oversee inter-agent communications, identify potential threats, enforce privacy and access controls, and maintain comprehensive audit records. Complementary to the idea of Sentinel Agents is the use of a Coordinator Agent. The Coordinator Agent supervises policy implementation, and manages agent participation. In addition, the Coordinator also ingests alerts from Sentinel Agents. Based on these alerts, it can adapt policies, isolate or quarantine misbehaving agents, and contain threats to maintain the integrity of the MAS ecosystem. This dual-layered security approach, combining the continuous monitoring of Sentinel Agents with the governance functions of Coordinator Agents, supports dynamic and adaptive defense mechanisms against a range of threats, including prompt injection, collusive agent behavior, hallucinations generated by LLMs, privacy breaches, and coordinated multi-agent attacks. In addition to the architectural design, we present a simulation study where 162 synthetic attacks of different families (prompt injection, hallucination, and data exfiltration) were injected into a multi-agent conversational environment. The Sentinel Agents successfully detected the attack attempts, confirming the practical feasibility of the proposed monitoring approach. The framework also offers enhanced system observability, supports regulatory compliance, and enables policy evolution over time.

摘要: 本文提出了一种新颖的架构框架，旨在增强多代理系统（MAS）的安全性和可靠性。该框架的核心组件是Sentinel Agents网络，充当分布式安全层，集成了通过大型语言模型（LLM）进行的语义分析、行为分析、检索增强验证和跨代理异常检测等技术。此类代理可以监督代理间的通信、识别潜在威胁、实施隐私和访问控制以及维护全面的审计记录。对哨兵代理想法的补充是协调代理的使用。协调员代理监督政策实施并管理代理参与。此外，协调员还接收来自哨兵特工的警报。基于这些警报，它可以调整政策、隔离或隔离行为不端的代理，并遏制威胁以维护MAS生态系统的完整性。这种双层安全方法将哨兵代理的持续监控与协调代理的治理功能相结合，支持针对一系列威胁的动态和自适应防御机制，包括即时注入、串通代理行为、LLM产生的幻觉、隐私泄露和协调多代理攻击。除了架构设计之外，我们还进行了一项模拟研究，其中将不同家庭的162种合成攻击（即时注射、幻觉和数据泄露）注入到多智能体对话环境中。哨兵特工成功检测到攻击企图，证实了拟议监控方法的实际可行性。该框架还提供增强的系统可观察性、支持监管合规性并使政策能够随着时间的推移而演变。



## **8. MUSE: MCTS-Driven Red Teaming Framework for Enhanced Multi-Turn Dialogue Safety in Large Language Models**

MUSE：MCTS驱动的红色团队框架，用于增强大型语言模型中的多回合对话安全性 cs.CL

EMNLP 2025 main conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14651v1) [paper-pdf](http://arxiv.org/pdf/2509.14651v1)

**Authors**: Siyu Yan, Long Zeng, Xuecheng Wu, Chengcheng Han, Kongcheng Zhang, Chong Peng, Xuezhi Cao, Xunliang Cai, Chenjuan Guo

**Abstract**: As large language models~(LLMs) become widely adopted, ensuring their alignment with human values is crucial to prevent jailbreaks where adversaries manipulate models to produce harmful content. While most defenses target single-turn attacks, real-world usage often involves multi-turn dialogues, exposing models to attacks that exploit conversational context to bypass safety measures. We introduce MUSE, a comprehensive framework tackling multi-turn jailbreaks from both attack and defense angles. For attacks, we propose MUSE-A, a method that uses frame semantics and heuristic tree search to explore diverse semantic trajectories. For defense, we present MUSE-D, a fine-grained safety alignment approach that intervenes early in dialogues to reduce vulnerabilities. Extensive experiments on various models show that MUSE effectively identifies and mitigates multi-turn vulnerabilities. Code is available at \href{https://github.com/yansiyu02/MUSE}{https://github.com/yansiyu02/MUSE}.

摘要: 随着大型语言模型（LLM）的广泛采用，确保它们与人类价值观保持一致对于防止对手操纵模型产生有害内容的越狱至关重要。虽然大多数防御措施都针对单轮攻击，但现实世界的使用通常涉及多轮对话，从而使模型暴露于利用对话上下文绕过安全措施的攻击中。我们引入了MUSE，这是一个从攻击和防御角度解决多回合越狱的综合框架。对于攻击，我们提出了MUE-A，这是一种使用框架语义和启发式树搜索来探索不同的语义轨迹的方法。对于防御，我们提出了MUE-D，这是一种细粒度的安全调整方法，可以在对话中早期干预以减少漏洞。对各种模型的广泛实验表明，MUSE可以有效识别和缓解多回合漏洞。代码可访问\href{https：//github.com/yansiyu02/MUSE}{https：//github.com/yansiyu02/MUSE}。



## **9. Enterprise AI Must Enforce Participant-Aware Access Control**

企业人工智能必须强制执行用户感知访问控制 cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14608v1) [paper-pdf](http://arxiv.org/pdf/2509.14608v1)

**Authors**: Shashank Shreedhar Bhatt, Tanmay Rajore, Khushboo Aggarwal, Ganesh Ananthanarayanan, Ranveer Chandra, Nishanth Chandran, Suyash Choudhury, Divya Gupta, Emre Kiciman, Sumit Kumar Pandey, Srinath Setty, Rahul Sharma, Teijia Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in enterprise settings where they interact with multiple users and are trained or fine-tuned on sensitive internal data. While fine-tuning enhances performance by internalizing domain knowledge, it also introduces a critical security risk: leakage of confidential training data to unauthorized users. These risks are exacerbated when LLMs are combined with Retrieval-Augmented Generation (RAG) pipelines that dynamically fetch contextual documents at inference time.   We demonstrate data exfiltration attacks on AI assistants where adversaries can exploit current fine-tuning and RAG architectures to leak sensitive information by leveraging the lack of access control enforcement. We show that existing defenses, including prompt sanitization, output filtering, system isolation, and training-level privacy mechanisms, are fundamentally probabilistic and fail to offer robust protection against such attacks.   We take the position that only a deterministic and rigorous enforcement of fine-grained access control during both fine-tuning and RAG-based inference can reliably prevent the leakage of sensitive data to unauthorized recipients.   We introduce a framework centered on the principle that any content used in training, retrieval, or generation by an LLM is explicitly authorized for \emph{all users involved in the interaction}. Our approach offers a simple yet powerful paradigm shift for building secure multi-user LLM systems that are grounded in classical access control but adapted to the unique challenges of modern AI workflows. Our solution has been deployed in Microsoft Copilot Tuning, a product offering that enables organizations to fine-tune models using their own enterprise-specific data.

摘要: 大型语言模型（LLM）越来越多地部署在企业环境中，它们与多个用户交互，并根据敏感的内部数据接受培训或微调。虽然微调通过内化领域知识来提高性能，但它也会带来严重的安全风险：机密培训数据泄露给未经授权的用户。当LLM与在推理时动态获取上下文文档的检索增强生成（RAG）管道相结合时，这些风险就会加剧。   我们展示了对人工智能助手的数据泄露攻击，其中对手可以利用当前的微调和RAG架构，通过利用访问控制强制执行的缺乏来泄露敏感信息。我们表明，现有的防御措施，包括即时清理、输出过滤、系统隔离和训练级隐私机制，从根本上来说是概率性的，无法提供针对此类攻击的强有力保护。   我们的立场是，只有在微调和基于RAG的推理期间确定性且严格地执行细粒度的访问控制，才能可靠地防止敏感数据泄露给未经授权的接收者。   我们引入了一个框架，其核心原则是LLM在训练、检索或生成中使用的任何内容都被明确授权给\{参与交互的所有用户}。我们的方法为构建安全的多用户LLM系统提供了简单而强大的范式转变，该系统基于经典的访问控制，但适应现代人工智能工作流程的独特挑战。我们的解决方案已部署在Microsoft Copilot Tuning中，这是一种产品，使组织能够使用自己的企业特定数据微调模型。



## **10. Reconstruction of Differentially Private Text Sanitization via Large Language Models**

通过大语言模型重建差异私人文本清理 cs.CR

RAID-2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2410.12443v3) [paper-pdf](http://arxiv.org/pdf/2410.12443v3)

**Authors**: Shuchao Pang, Zhigang Lu, Haichen Wang, Peng Fu, Yongbin Zhou, Minhui Xue

**Abstract**: Differential privacy (DP) is the de facto privacy standard against privacy leakage attacks, including many recently discovered ones against large language models (LLMs). However, we discovered that LLMs could reconstruct the altered/removed privacy from given DP-sanitized prompts. We propose two attacks (black-box and white-box) based on the accessibility to LLMs and show that LLMs could connect the pair of DP-sanitized text and the corresponding private training data of LLMs by giving sample text pairs as instructions (in the black-box attacks) or fine-tuning data (in the white-box attacks). To illustrate our findings, we conduct comprehensive experiments on modern LLMs (e.g., LLaMA-2, LLaMA-3, ChatGPT-3.5, ChatGPT-4, ChatGPT-4o, Claude-3, Claude-3.5, OPT, GPT-Neo, GPT-J, Gemma-2, and Pythia) using commonly used datasets (such as WikiMIA, Pile-CC, and Pile-Wiki) against both word-level and sentence-level DP. The experimental results show promising recovery rates, e.g., the black-box attacks against the word-level DP over WikiMIA dataset gave 72.18% on LLaMA-2 (70B), 82.39% on LLaMA-3 (70B), 75.35% on Gemma-2, 91.2% on ChatGPT-4o, and 94.01% on Claude-3.5 (Sonnet). More urgently, this study indicates that these well-known LLMs have emerged as a new security risk for existing DP text sanitization approaches in the current environment.

摘要: 差异隐私（DP）是针对隐私泄露攻击的事实上的隐私标准，包括最近发现的许多针对大型语言模型（LLM）的攻击。然而，我们发现LLM可以从给定的DP消毒提示中重建更改/删除的隐私。我们基于LLM的可访问性提出了两种攻击（黑匣子和白盒），并表明LLM可以通过提供样本文本对作为指令（在黑匣子攻击中）或微调数据（在白盒攻击中）来连接DP清理文本对和LLM的相应私人训练数据。为了说明我们的发现，我们对现代LLM进行了全面的实验（例如，LLaMA-2、LLaMA-3、ChatGPT-3.5、ChatGPT-4、ChatGPT-4 o、Claude-3、Claude-3.5、OPT、GPT-Neo、GPT-J、Gemma-2和Pythia）针对单词级和业务级DP使用常用数据集（例如WikiMIA、Pile-CC和Pile-iki）。实验结果显示出有希望的回收率，例如，针对WikiMIA数据集的词级DP的黑匣子攻击在LLaMA-2（70 B）上为72.18%，在LLaMA-3（70 B）上为82.39%，在Gemma-2上为75.35%，在ChatGPT-4 o上为91.2%，在Claude-3.5（十四行诗）上为94.01%。更紧迫的是，这项研究表明，这些著名的LLM已成为当前环境下现有DP文本清理方法的新安全风险。



## **11. SynBench: A Benchmark for Differentially Private Text Generation**

SynBench：差异私密文本生成的基准 cs.AI

15 pages

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14594v1) [paper-pdf](http://arxiv.org/pdf/2509.14594v1)

**Authors**: Yidan Sun, Viktor Schlegel, Srinivasan Nandakumar, Iqra Zahid, Yuping Wu, Yulong Wu, Hao Li, Jie Zhang, Warren Del-Pinto, Goran Nenadic, Siew Kei Lam, Anil Anthony Bharath

**Abstract**: Data-driven decision support in high-stakes domains like healthcare and finance faces significant barriers to data sharing due to regulatory, institutional, and privacy concerns. While recent generative AI models, such as large language models, have shown impressive performance in open-domain tasks, their adoption in sensitive environments remains limited by unpredictable behaviors and insufficient privacy-preserving datasets for benchmarking. Existing anonymization methods are often inadequate, especially for unstructured text, as redaction and masking can still allow re-identification. Differential Privacy (DP) offers a principled alternative, enabling the generation of synthetic data with formal privacy assurances. In this work, we address these challenges through three key contributions. First, we introduce a comprehensive evaluation framework with standardized utility and fidelity metrics, encompassing nine curated datasets that capture domain-specific complexities such as technical jargon, long-context dependencies, and specialized document structures. Second, we conduct a large-scale empirical study benchmarking state-of-the-art DP text generation methods and LLMs of varying sizes and different fine-tuning strategies, revealing that high-quality domain-specific synthetic data generation under DP constraints remains an unsolved challenge, with performance degrading as domain complexity increases. Third, we develop a membership inference attack (MIA) methodology tailored for synthetic text, providing first empirical evidence that the use of public datasets - potentially present in pre-training corpora - can invalidate claimed privacy guarantees. Our findings underscore the urgent need for rigorous privacy auditing and highlight persistent gaps between open-domain and specialist evaluations, informing responsible deployment of generative AI in privacy-sensitive, high-stakes settings.

摘要: 由于监管、机构和隐私问题，医疗保健和金融等高风险领域的数据驱动决策支持在数据共享方面面临巨大障碍。虽然最近的生成性人工智能模型（例如大型语言模型）在开放领域任务中表现出了令人印象深刻的性能，但它们在敏感环境中的采用仍然受到不可预测的行为和用于基准测试的隐私保护数据集不足的限制。现有的匿名化方法通常不充分，尤其是对于非结构化文本，因为编辑和掩蔽仍然可以允许重新识别。差异隐私（DP）提供了一种有原则的替代方案，可以生成具有正式隐私保证的合成数据。在这项工作中，我们通过三项关键贡献来应对这些挑战。首先，我们引入了一个具有标准化效用和保真度指标的全面评估框架，其中包含九个精心策划的数据集，这些数据集捕捉特定领域的复杂性，例如技术行话、长上下文依赖性和专业文档结构。其次，我们进行了一项大规模的实证研究，对最先进的DP文本生成方法和不同规模和不同微调策略的LLM进行了基准测试，揭示了DP约束下的高质量特定领域的合成数据生成仍然是一个尚未解决的挑战，随着领域复杂性的增加，性能会下降。第三，我们开发了一种专为合成文本量身定制的成员资格推理攻击（MIA）方法，提供了第一个经验证据，证明使用公共数据集（可能存在于预训练库中）可以使声称的隐私保证无效。我们的研究结果强调了严格的隐私审计的迫切需要，并强调了开放领域和专业评估之间持续存在的差距，为在隐私敏感、高风险的环境中负责任地部署生成性人工智能提供信息。



## **12. LLM Jailbreak Detection for (Almost) Free!**

LLM越狱检测（几乎）免费！ cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14558v1) [paper-pdf](http://arxiv.org/pdf/2509.14558v1)

**Authors**: Guorui Chen, Yifan Xia, Xiaojun Jia, Zhijiang Li, Philip Torr, Jindong Gu

**Abstract**: Large language models (LLMs) enhance security through alignment when widely used, but remain susceptible to jailbreak attacks capable of producing inappropriate content. Jailbreak detection methods show promise in mitigating jailbreak attacks through the assistance of other models or multiple model inferences. However, existing methods entail significant computational costs. In this paper, we first present a finding that the difference in output distributions between jailbreak and benign prompts can be employed for detecting jailbreak prompts. Based on this finding, we propose a Free Jailbreak Detection (FJD) which prepends an affirmative instruction to the input and scales the logits by temperature to further distinguish between jailbreak and benign prompts through the confidence of the first token. Furthermore, we enhance the detection performance of FJD through the integration of virtual instruction learning. Extensive experiments on aligned LLMs show that our FJD can effectively detect jailbreak prompts with almost no additional computational costs during LLM inference.

摘要: 大型语言模型（LLM）在广泛使用时通过对齐来增强安全性，但仍然容易受到能够产生不当内容的越狱攻击。越狱检测方法在通过其他模型或多个模型推断的帮助减轻越狱攻击方面表现出了希望。然而，现有方法需要大量的计算成本。在本文中，我们首先提出了一个发现，即越狱和良性提示之间的输出分布差异可以用于检测越狱提示。基于这一发现，我们提出了一种免费越狱检测（FJD），它在输入中预先添加肯定指令，并通过温度缩放逻辑比特，以通过第一个令牌的置信度进一步区分越狱和良性提示。此外，我们还通过集成虚拟教学学习来提高FJD的检测性能。对对齐LLM的大量实验表明，我们的FJD可以有效地检测越狱提示，而在LLM推理期间几乎没有额外的计算成本。



## **13. GRADA: Graph-based Reranking against Adversarial Documents Attack**

GRADA：基于图的重新排名对抗文档攻击 cs.IR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2505.07546v3) [paper-pdf](http://arxiv.org/pdf/2505.07546v3)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.

摘要: 检索增强生成（RAG）框架通过集成来自检索文档的外部知识来提高大型语言模型（LLM）的准确性，从而克服模型静态内在知识的局限性。然而，这些系统很容易受到对抗性攻击，这些攻击通过引入对抗性但在语义上与查询相似的文档来操纵检索过程。值得注意的是，虽然这些对抗性文档类似于查询，但它们与检索集中的良性文档表现出弱的相似性。因此，我们提出了一个简单而有效的基于图形的对抗性文档攻击重新排名（GRADA）框架，旨在保留检索质量，同时显着降低对手的成功。我们的研究通过在五个LLM上进行的实验来评估我们的方法的有效性：GPT-3.5-Turbo，GPT-4 o，Llama3.1-8b，Llama3.1- 70 b和Qwen2.5- 7 b。我们使用三个数据集来评估性能，来自Natural Questions数据集的结果表明攻击成功率降低了80%，同时保持了最小的准确性损失。



## **14. Benchmarking Large Language Models for Cryptanalysis and Side-Channel Vulnerabilities**

针对加密分析和侧通道漏洞对大型语言模型进行基准测试 cs.CL

EMNLP'25 Findings

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2505.24621v2) [paper-pdf](http://arxiv.org/pdf/2505.24621v2)

**Authors**: Utsav Maskey, Chencheng Zhu, Usman Naseem

**Abstract**: Recent advancements in large language models (LLMs) have transformed natural language understanding and generation, leading to extensive benchmarking across diverse tasks. However, cryptanalysis - a critical area for data security and its connection to LLMs' generalization abilities - remains underexplored in LLM evaluations. To address this gap, we evaluate the cryptanalytic potential of state-of-the-art LLMs on ciphertexts produced by a range of cryptographic algorithms. We introduce a benchmark dataset of diverse plaintexts, spanning multiple domains, lengths, writing styles, and topics, paired with their encrypted versions. Using zero-shot and few-shot settings along with chain-of-thought prompting, we assess LLMs' decryption success rate and discuss their comprehension abilities. Our findings reveal key insights into LLMs' strengths and limitations in side-channel scenarios and raise concerns about their susceptibility to under-generalization-related attacks. This research highlights the dual-use nature of LLMs in security contexts and contributes to the ongoing discussion on AI safety and security.

摘要: 大型语言模型（LLM）的最新进展已经改变了自然语言的理解和生成，导致了跨各种任务的广泛基准测试。然而，密码分析-数据安全的一个关键领域及其与LLM泛化能力的联系-在LLM评估中仍然没有得到充分的探索。为了解决这一差距，我们评估的密码分析潜力的国家的最先进的LLM的密文产生的一系列密码算法。我们介绍了一个基准数据集的不同明文，跨越多个领域，长度，写作风格和主题，与他们的加密版本配对。使用零镜头和少镜头设置以及思想链提示，我们评估LLM的解密成功率并讨论他们的理解能力。我们的研究结果揭示了对LLM在侧通道场景中的优势和局限性的关键见解，并引发了人们对它们容易受到归因不足相关攻击的担忧。这项研究强调了LLM在安全环境中的双重用途性质，并有助于正在进行的关于人工智能安全性的讨论。



## **15. Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs**

评估和改进LLM生成的安全攻击检测器的鲁棒性 cs.SE

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2411.18216v2) [paper-pdf](http://arxiv.org/pdf/2411.18216v2)

**Authors**: Samuele Pasini, Jinhan Kim, Tommaso Aiello, Rocio Cabrera Lozoya, Antonino Sabetta, Paolo Tonella

**Abstract**: Large Language Models (LLMs) are increasingly used in software development to generate functions, such as attack detectors, that implement security requirements. A key challenge is ensuring the LLMs have enough knowledge to address specific security requirements, such as information about existing attacks. For this, we propose an approach integrating Retrieval Augmented Generation (RAG) and Self-Ranking into the LLM pipeline. RAG enhances the robustness of the output by incorporating external knowledge sources, while the Self-Ranking technique, inspired by the concept of Self-Consistency, generates multiple reasoning paths and creates ranks to select the most robust detector. Our extensive empirical study targets code generated by LLMs to detect two prevalent injection attacks in web security: Cross-Site Scripting (XSS) and SQL injection (SQLi). Results show a significant improvement in detection performance while employing RAG and Self-Ranking, with an increase of up to 71%pt (on average 37%pt) and up to 43%pt (on average 6%pt) in the F2-Score for XSS and SQLi detection, respectively.

摘要: 大型语言模型（LLM）越来越多地用于软件开发来生成实现安全要求的函数，例如攻击检测器。一个关键挑战是确保LLM拥有足够的知识来满足特定的安全要求，例如有关现有攻击的信息。为此，我们提出了一种将检索增强生成（RAG）和自我排名集成到LLM管道中的方法。RAG通过整合外部知识源来增强输出的稳健性，而自排名技术则受到自一致性概念的启发，生成多个推理路径并创建排名来选择最稳健的检测器。我们广泛的实证研究针对LLM生成的代码，以检测网络安全中两种普遍的注入攻击：跨站点脚本（XSS）和SQL注入（SQLi）。结果显示，采用RAG和Self-Ranking时检测性能显着提高，XSS和SQLi检测的F2评分分别增加了71%pt（平均37%pt）和43%pt（平均6%pt）。



## **16. CyberLLMInstruct: A Pseudo-malicious Dataset Revealing Safety-performance Trade-offs in Cyber Security LLM Fine-tuning**

CyberLLMDirecct：揭示网络安全中安全性能权衡的伪恶意数据集LLM微调 cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2503.09334v3) [paper-pdf](http://arxiv.org/pdf/2503.09334v3)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents both opportunities and critical safety risks. We introduce CyberLLMInstruct, a dataset of 54,928 pseudo-malicious instruction-response pairs spanning cyber security tasks including malware analysis, phishing simulations, and zero-day vulnerabilities. Our comprehensive evaluation using seven open-source LLMs reveals a critical trade-off: while fine-tuning improves cyber security task performance (achieving up to 92.50% accuracy on CyberMetric), it severely compromises safety resilience across all tested models and attack vectors (e.g., Llama 3.1 8B's security score against prompt injection drops from 0.95 to 0.15). The dataset incorporates diverse sources including CTF challenges, academic papers, industry reports, and CVE databases to ensure comprehensive coverage of cyber security domains. Our findings highlight the unique challenges of securing LLMs in adversarial domains and establish the critical need for developing fine-tuning methodologies that balance performance gains with safety preservation in security-sensitive domains.

摘要: 将大型语言模型（LLM）集成到网络安全应用程序中既带来了机遇，也带来了严重的安全风险。我们引入CyberLLMCinsert，这是一个由54，928个伪恶意描述-响应对组成的数据集，涵盖网络安全任务，包括恶意软件分析、网络钓鱼模拟和零日漏洞。我们使用七个开源LLM进行的全面评估揭示了一个关键的权衡：虽然微调可以提高网络安全任务性能（在CyberMetric上实现高达92.50%的准确率），但它严重损害了所有测试模型和攻击载体的安全弹性（例如，Lama 3.1 8B对立即注射的安全评分从0.95下降到0.15）。该数据集融合了多种来源，包括CTF挑战、学术论文、行业报告和UTE数据库，以确保网络安全领域的全面覆盖。我们的研究结果强调了在对抗性领域中保护LLM的独特挑战，并确定了开发微调方法的迫切需求，该方法在安全敏感领域中平衡性能收益与安全保护。



## **17. Do LLMs Align Human Values Regarding Social Biases? Judging and Explaining Social Biases with LLMs**

法学硕士是否在社会偏见方面与人类价值观保持一致？利用法学硕士判断和解释社会偏见 cs.CL

38 pages, 31 figures

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.13869v1) [paper-pdf](http://arxiv.org/pdf/2509.13869v1)

**Authors**: Yang Liu, Chenhui Chu

**Abstract**: Large language models (LLMs) can lead to undesired consequences when misaligned with human values, especially in scenarios involving complex and sensitive social biases. Previous studies have revealed the misalignment of LLMs with human values using expert-designed or agent-based emulated bias scenarios. However, it remains unclear whether the alignment of LLMs with human values differs across different types of scenarios (e.g., scenarios containing negative vs. non-negative questions). In this study, we investigate the alignment of LLMs with human values regarding social biases (HVSB) in different types of bias scenarios. Through extensive analysis of 12 LLMs from four model families and four datasets, we demonstrate that LLMs with large model parameter scales do not necessarily have lower misalignment rate and attack success rate. Moreover, LLMs show a certain degree of alignment preference for specific types of scenarios and the LLMs from the same model family tend to have higher judgment consistency. In addition, we study the understanding capacity of LLMs with their explanations of HVSB. We find no significant differences in the understanding of HVSB across LLMs. We also find LLMs prefer their own generated explanations. Additionally, we endow smaller language models (LMs) with the ability to explain HVSB. The generation results show that the explanations generated by the fine-tuned smaller LMs are more readable, but have a relatively lower model agreeability.

摘要: 大型语言模型（LLM）与人类价值观不一致时可能会导致不良后果，尤其是在涉及复杂和敏感的社会偏见的场景中。之前的研究使用专家设计或基于代理的模拟偏见场景揭示了LLM与人类价值观的不一致。然而，目前尚不清楚LLM与人类价值观的一致是否在不同类型的场景中有所不同（例如，包含负面问题与非负面问题的场景）。在这项研究中，我们调查了在不同类型的偏见场景中，LLM与人类社会偏见（HCSB）价值观的一致性。通过对来自四个模型系列和四个数据集的12个LLM的广泛分析，我们证明具有大模型参数规模的LLM不一定具有较低的失准率和攻击成功率。此外，LLM对特定类型的场景表现出一定程度的一致偏好，并且来自同一模型家族的LLM往往具有更高的判断一致性。此外，我们还研究了LLM的理解能力及其对HCSB的解释。我们发现各LLM对HCSB的理解没有显着差异。我们还发现LLM更喜欢他们自己生成的解释。此外，我们赋予较小的语言模型（LM）解释HDSB的能力。生成结果表明，微调后的较小LM生成的解释更具可读性，但具有相对较低的模型。



## **18. Defending against Indirect Prompt Injection by Instruction Detection**

利用指令检测防御间接提示注入 cs.CR

16 pages, 4 figures

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2505.06311v2) [paper-pdf](http://arxiv.org/pdf/2505.06311v2)

**Authors**: Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, Fangzhao Wu

**Abstract**: The integration of Large Language Models (LLMs) with external sources is becoming increasingly common, with Retrieval-Augmented Generation (RAG) being a prominent example. However, this integration introduces vulnerabilities of Indirect Prompt Injection (IPI) attacks, where hidden instructions embedded in external data can manipulate LLMs into executing unintended or harmful actions. We recognize that IPI attacks fundamentally rely on the presence of instructions embedded within external content, which can alter the behavioral states of LLMs. Can the effective detection of such state changes help us defend against IPI attacks? In this paper, we propose InstructDetector, a novel detection-based approach that leverages the behavioral states of LLMs to identify potential IPI attacks. Specifically, we demonstrate the hidden states and gradients from intermediate layers provide highly discriminative features for instruction detection. By effectively combining these features, InstructDetector achieves a detection accuracy of 99.60% in the in-domain setting and 96.90% in the out-of-domain setting, and reduces the attack success rate to just 0.03% on the BIPIA benchmark. The code is publicly available at https://github.com/MYVAE/Instruction-detection.

摘要: 大型语言模型（LLM）与外部源的集成变得越来越常见，检索增强生成（RAG）就是一个突出的例子。然而，此集成引入了间接提示注入（IPI）攻击的漏洞，其中嵌入外部数据中的隐藏指令可以操纵LLM执行无意或有害的操作。我们认识到，IPI攻击从根本上依赖于外部内容中嵌入的指令的存在，这些指令可以改变LLM的行为状态。有效检测此类状态变化能否帮助我们抵御IPI攻击？在本文中，我们提出了DirectDetector，这是一种新型的基于检测的方法，利用LLM的行为状态来识别潜在的IPI攻击。具体来说，我们证明了来自中间层的隐藏状态和梯度为指令检测提供了高度区分性的特征。通过有效结合这些功能，DirectDetector在域内设置中的检测准确率为99.60%，在域外设置中的检测准确率为96.90%，并将BIPIA基准测试中的攻击成功率降低至仅0.03%。该代码可在https://github.com/MYVAE/Instruction-detection上公开获取。



## **19. DiffHash: Text-Guided Targeted Attack via Diffusion Models against Deep Hashing Image Retrieval**

迪夫哈希：通过针对深度哈希图像检索的扩散模型进行文本引导定向攻击 cs.IR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.12824v2) [paper-pdf](http://arxiv.org/pdf/2509.12824v2)

**Authors**: Zechao Liu, Zheng Zhou, Xiangkun Chen, Tao Liang, Dapeng Lang

**Abstract**: Deep hashing models have been widely adopted to tackle the challenges of large-scale image retrieval. However, these approaches face serious security risks due to their vulnerability to adversarial examples. Despite the increasing exploration of targeted attacks on deep hashing models, existing approaches still suffer from a lack of multimodal guidance, reliance on labeling information and dependence on pixel-level operations for attacks. To address these limitations, we proposed DiffHash, a novel diffusion-based targeted attack for deep hashing. Unlike traditional pixel-based attacks that directly modify specific pixels and lack multimodal guidance, our approach focuses on optimizing the latent representations of images, guided by text information generated by a Large Language Model (LLM) for the target image. Furthermore, we designed a multi-space hash alignment network to align the high-dimension image space and text space to the low-dimension binary hash space. During reconstruction, we also incorporated text-guided attention mechanisms to refine adversarial examples, ensuring them aligned with the target semantics while maintaining visual plausibility. Extensive experiments have demonstrated that our method outperforms state-of-the-art (SOTA) targeted attack methods, achieving better black-box transferability and offering more excellent stability across datasets.

摘要: 深度哈希模型已被广泛采用来应对大规模图像检索的挑战。然而，由于这些方法容易受到对抗示例的影响，因此面临严重的安全风险。尽管人们越来越多地探索深度哈希模型的有针对性的攻击，但现有方法仍然缺乏多模式指导、依赖标签信息以及依赖像素级操作进行攻击。为了解决这些限制，我们提出了迪夫哈希，这是一种新型的基于扩散的深度哈希定向攻击。与直接修改特定像素且缺乏多模式指导的传统基于像素的攻击不同，我们的方法重点是优化图像的潜在表示，并由目标图像的大型语言模型（LLM）生成的文本信息指导。此外，我们设计了一个多空间哈希对齐网络，将多维图像空间和文本空间与低维二进制哈希空间对齐。在重建过程中，我们还结合了文本引导的注意力机制来完善对抗性示例，确保它们与目标语义保持一致，同时保持视觉可信性。大量实验表明，我们的方法优于最先进的（SOTA）定向攻击方法，实现了更好的黑匣子可转移性，并在数据集之间提供更出色的稳定性。



## **20. Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in Retrieval-Augmented Generation**

谁撒了谎？检索增强一代有毒知识的责任归因 cs.CR

To appear in the IEEE Symposium on Security and Privacy, 2026

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.13772v1) [paper-pdf](http://arxiv.org/pdf/2509.13772v1)

**Authors**: Baolei Zhang, Haoran Xin, Yuxi Chen, Zhuqing Liu, Biao Yi, Tong Li, Lihai Nie, Zheli Liu, Minghong Fang

**Abstract**: Retrieval-Augmented Generation (RAG) integrates external knowledge into large language models to improve response quality. However, recent work has shown that RAG systems are highly vulnerable to poisoning attacks, where malicious texts are inserted into the knowledge database to influence model outputs. While several defenses have been proposed, they are often circumvented by more adaptive or sophisticated attacks.   This paper presents RAGOrigin, a black-box responsibility attribution framework designed to identify which texts in the knowledge database are responsible for misleading or incorrect generations. Our method constructs a focused attribution scope tailored to each misgeneration event and assigns a responsibility score to each candidate text by evaluating its retrieval ranking, semantic relevance, and influence on the generated response. The system then isolates poisoned texts using an unsupervised clustering method. We evaluate RAGOrigin across seven datasets and fifteen poisoning attacks, including newly developed adaptive poisoning strategies and multi-attacker scenarios. Our approach outperforms existing baselines in identifying poisoned content and remains robust under dynamic and noisy conditions. These results suggest that RAGOrigin provides a practical and effective solution for tracing the origins of corrupted knowledge in RAG systems.

摘要: 检索增强生成（RAG）将外部知识集成到大型语言模型中，以提高响应质量。然而，最近的工作表明，RAG系统非常容易受到中毒攻击，恶意文本被插入到知识数据库中以影响模型输出。虽然已经提出了多种防御措施，但它们通常会被更具适应性或复杂性的攻击所规避。   本文提出了RAGNi，这是一个黑匣子责任归因框架，旨在识别知识数据库中的哪些文本应对误导性或错误的生成负责。我们的方法构建了一个针对每个错误生成事件量身定制的有针对性的归因范围，并通过评估每个候选文本的检索排名、语义相关性和对生成的响应的影响来为每个候选文本分配责任分数。然后，该系统使用无监督集群方法隔离有毒文本。我们对7个数据集和15个中毒攻击进行了评估，包括新开发的自适应中毒策略和多攻击者场景。我们的方法在识别有毒内容方面优于现有基线，并且在动态和噪音条件下保持稳健。这些结果表明，RAGNY为追踪RAG系统中损坏知识的起源提供了一种实用有效的解决方案。



## **21. A Simple and Efficient Jailbreak Method Exploiting LLMs' Helpfulness**

一种简单有效的越狱方法，利用LL M的帮助 cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.14297v1) [paper-pdf](http://arxiv.org/pdf/2509.14297v1)

**Authors**: Xuan Luo, Yue Wang, Zefeng He, Geng Tu, Jing Li, Ruifeng Xu

**Abstract**: Safety alignment aims to prevent Large Language Models (LLMs) from responding to harmful queries. To strengthen safety protections, jailbreak methods are developed to simulate malicious attacks and uncover vulnerabilities. In this paper, we introduce HILL (Hiding Intention by Learning from LLMs), a novel jailbreak approach that systematically transforms imperative harmful requests into learning-style questions with only straightforward hypotheticality indicators. Further, we introduce two new metrics to thoroughly evaluate the utility of jailbreak methods. Experiments on the AdvBench dataset across a wide range of models demonstrate HILL's strong effectiveness, generalizability, and harmfulness. It achieves top attack success rates on the majority of models and across malicious categories while maintaining high efficiency with concise prompts. Results of various defense methods show the robustness of HILL, with most defenses having mediocre effects or even increasing the attack success rates. Moreover, the assessment on our constructed safe prompts reveals inherent limitations of LLMs' safety mechanisms and flaws in defense methods. This work exposes significant vulnerabilities of safety measures against learning-style elicitation, highlighting a critical challenge of balancing helpfulness and safety alignments.

摘要: 安全调整旨在防止大型语言模型（LLM）响应有害查询。为了加强安全保护，开发了越狱方法来模拟恶意攻击并发现漏洞。在本文中，我们介绍了HILL（通过从LLM学习来隐藏意图），这是一种新颖的越狱方法，它系统地将强制性有害请求转化为仅具有简单假设性指标的学习式问题。此外，我们引入了两个新的指标来彻底评估越狱方法的实用性。在AdvBench数据集上进行的各种模型的实验证明了HILL的强大有效性、可概括性和危害性。它在大多数模型和恶意类别上实现了最高的攻击成功率，同时通过简洁的提示保持高效率。各种防御方法的结果表明HILL的稳健性，大多数防御效果平平，甚至提高攻击成功率。此外，对我们构建的安全提示的评估揭示了LLM安全机制的固有局限性和防御方法的缺陷。这项工作暴露了安全措施针对学习式启发的显着弱点，凸显了平衡帮助性和安全性的关键挑战。



## **22. SoK: How Sensor Attacks Disrupt Autonomous Vehicles: An End-to-end Analysis, Challenges, and Missed Threats**

SoK：传感器攻击如何扰乱自动驾驶车辆：端到端分析、挑战和错过的威胁 cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.11120v3) [paper-pdf](http://arxiv.org/pdf/2509.11120v3)

**Authors**: Qingzhao Zhang, Shaocheng Luo, Z. Morley Mao, Miroslav Pajic, Michael K. Reiter

**Abstract**: Autonomous vehicles, including self-driving cars, robotic ground vehicles, and drones, rely on complex sensor pipelines to ensure safe and reliable operation. However, these safety-critical systems remain vulnerable to adversarial sensor attacks that can compromise their performance and mission success. While extensive research has demonstrated various sensor attack techniques, critical gaps remain in understanding their feasibility in real-world, end-to-end systems. This gap largely stems from the lack of a systematic perspective on how sensor errors propagate through interconnected modules in autonomous systems when autonomous vehicles interact with the physical world.   To bridge this gap, we present a comprehensive survey of autonomous vehicle sensor attacks across platforms, sensor modalities, and attack methods. Central to our analysis is the System Error Propagation Graph (SEPG), a structured demonstration tool that illustrates how sensor attacks propagate through system pipelines, exposing the conditions and dependencies that determine attack feasibility. With the aid of SEPG, our study distills seven key findings that highlight the feasibility challenges of sensor attacks and uncovers eleven previously overlooked attack vectors exploiting inter-module interactions, several of which we validate through proof-of-concept experiments. Additionally, we demonstrate how large language models (LLMs) can automate aspects of SEPG construction and cross-validate expert analysis, showcasing the promise of AI-assisted security evaluation.

摘要: 自动驾驶汽车、机器人地面车辆和无人机等自动驾驶车辆依赖复杂的传感器管道来确保安全可靠的运行。然而，这些安全关键系统仍然容易受到对抗性传感器攻击，这可能会损害其性能和任务成功。虽然广泛的研究已经证明了各种传感器攻击技术，但在了解其在现实世界的端到端系统中的可行性方面仍然存在重大差距。这一差距很大程度上源于缺乏对自动驾驶汽车与物理世界互动时传感器误差如何通过自动驾驶系统中的互连模块传播的系统视角。   为了弥合这一差距，我们对跨平台、传感器模式和攻击方法的自动驾驶汽车传感器攻击进行了全面调查。我们分析的核心是系统错误传播图（SEPG），这是一种结构化演示工具，它说明了传感器攻击如何通过系统管道传播，揭示了决定攻击可行性的条件和依赖性。在SEPG的帮助下，我们的研究提炼了七个关键发现，这些发现凸显了传感器攻击的可行性挑战，并揭示了11个以前被忽视的利用模块间交互的攻击载体，其中一些我们通过概念验证实验进行了验证。此外，我们还展示了大型语言模型（LLM）如何自动化SEPG构建的各个方面和交叉验证专家分析，展示了人工智能辅助安全评估的前景。



## **23. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

将逻辑与自身对立：通过对比问题探索模型辩护 cs.CL

Accepted at EMNLP 2025 (Main)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2501.01872v5) [paper-pdf](http://arxiv.org/pdf/2501.01872v5)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.

摘要: 尽管大型语言模型与人类价值观和道德原则广泛一致，但仍然容易受到利用其推理能力的复杂越狱攻击。现有的安全措施通常检测到明显的恶意意图，但无法解决微妙的、推理驱动的漏洞。在这项工作中，我们引入了POATE（极反相查询生成、对抗模板构建和搜索），这是一种新颖的越狱技术，利用对比推理来引发不道德的反应。POATE精心设计了语义上相反的意图，并将它们与对抗模板集成，以非凡的微妙性引导模型走向有害的输出。我们对参数大小不同的六个不同语言模型家族进行了广泛的评估，以证明攻击的稳健性，与现有方法相比，实现了显着更高的攻击成功率（~44%）。为了解决这个问题，我们提出了意图感知CoT和反向思维CoT，它们分解查询以检测恶意意图，并反向推理以评估和拒绝有害响应。这些方法增强了推理的稳健性并加强了模型对对抗性利用的防御。



## **24. From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing**

从能力到性能：在渗透测试中评估LLM架构的关键功能属性 cs.AI

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.14289v1) [paper-pdf](http://arxiv.org/pdf/2509.14289v1)

**Authors**: Lanxiao Huang, Daksh Dave, Ming Jin, Tyler Cody, Peter Beling

**Abstract**: Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.

摘要: 大型语言模型（LLM）越来越多地用于自动化或增强渗透测试，但它们在攻击阶段的有效性和可靠性仍不清楚。我们在现实的渗透测试场景中对多个基于LLM的代理（从单代理到模块化设计）进行了全面评估，测量经验性能和反复出现的故障模式。我们还通过有针对性的增强来隔离五种核心功能能力的影响：全球上下文记忆（GCM）、代理间消息传递（ILM）、上下文条件调用（CI）、自适应规划（AP）和实时监控（RTI）。这些干预措施分别支持：（i）上下文一致性和保留，（ii）组件间协调和状态管理，（iii）工具使用准确性和选择性执行，（iv）多步骤战略规划、错误检测和恢复，以及（v）实时动态响应能力。我们的结果表明，虽然一些架构本身表现出这些属性的子集，但有针对性的增强可以大大提高模块化代理的性能，特别是在复杂、多步骤和实时渗透测试任务中。



## **25. AQUA-LLM: Evaluating Accuracy, Quantization, and Adversarial Robustness Trade-offs in LLMs for Cybersecurity Question Answering**

AQUA-LLM：评估LLM中网络安全问题解答的准确性、量化和对抗性鲁棒性权衡 cs.CR

Accepted by the 24th IEEE International Conference on Machine  Learning and Applications (ICMLA'25)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13514v1) [paper-pdf](http://arxiv.org/pdf/2509.13514v1)

**Authors**: Onat Gungor, Roshan Sood, Harold Wang, Tajana Rosing

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong potential for cybersecurity question answering (QA), supporting decision-making in real-time threat detection and response workflows. However, their substantial computational demands pose significant challenges for deployment on resource-constrained edge devices. Quantization, a widely adopted model compression technique, can alleviate these constraints. Nevertheless, quantization may degrade model accuracy and increase susceptibility to adversarial attacks. Fine-tuning offers a potential means to mitigate these limitations, but its effectiveness when combined with quantization remains insufficiently explored. Hence, it is essential to understand the trade-offs among accuracy, efficiency, and robustness. We propose AQUA-LLM, an evaluation framework designed to benchmark several state-of-the-art small LLMs under four distinct configurations: base, quantized-only, fine-tuned, and fine-tuned combined with quantization, specifically for cybersecurity QA. Our results demonstrate that quantization alone yields the lowest accuracy and robustness despite improving efficiency. In contrast, combining quantization with fine-tuning enhances both LLM robustness and predictive performance, achieving an optimal balance of accuracy, robustness, and efficiency. These findings highlight the critical need for quantization-aware, robustness-preserving fine-tuning methodologies to enable the robust and efficient deployment of LLMs for cybersecurity QA.

摘要: 大型语言模型（LLM）最近在网络安全问题回答（QA）方面展示了强大的潜力，支持实时威胁检测和响应工作流程中的决策。然而，它们巨大的计算需求对资源受限的边缘设备上的部署构成了重大挑战。量化是一种广泛采用的模型压缩技术，可以缓解这些限制。然而，量化可能会降低模型准确性并增加对对抗攻击的敏感性。微调提供了一种潜在的手段，以减轻这些限制，但其有效性时，结合量化仍然没有得到充分的探讨。因此，了解准确性、效率和鲁棒性之间的权衡至关重要。我们提出了AQUA-LLM，这是一个评估框架，旨在对四种不同配置下的几种最先进的小型LLM进行基准测试：基础，仅量化，微调和微调与量化相结合，专门用于网络安全QA。我们的研究结果表明，量化单独产生最低的准确性和鲁棒性，尽管提高了效率。相比之下，量化与微调相结合可以增强LLM稳健性和预测性能，实现准确性、稳健性和效率的最佳平衡。这些发现凸显了对量化感知、鲁棒性保持微调方法的迫切需求，以实现网络安全QA的LLM稳健、高效的部署。



## **26. A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks**

一种基于多Agent的LLM快速注入攻击防御流水线 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.14285v1) [paper-pdf](http://arxiv.org/pdf/2509.14285v1)

**Authors**: S M Asif Hossain, Ruksat Khan Shayoni, Mohd Ruhul Ameen, Akif Islam, M. F. Mridha, Jungpil Shin

**Abstract**: Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.

摘要: 提示注入攻击是大型语言模型（LLM）部署中的一个主要漏洞，用户输入中嵌入的恶意指令可以覆盖系统提示并引发意外行为。本文提出了一种新型的多代理防御框架，该框架在协调管道中使用专门的LLM代理来实时检测和抵消即时注入攻击。我们使用两种不同的架构来评估我们的方法：顺序代理链管道和基于分层协调器的系统。我们对两个LLM平台（ChatGLM和Llama 2）上的55种独特的即时注入攻击（分为8类，总共400个攻击实例）进行了全面评估，展示了显着的安全改进。在没有防御机制的情况下，ChatGLM的基线攻击成功率（ASB）达到30%，Llama 2的基线攻击成功率（ASB）达到20%。我们的多代理管道实现了100%的缓解，在所有测试场景中将ASB降低至0%。该框架展示了多种攻击类别的稳健性，包括直接覆盖、代码执行尝试、数据溢出和模糊技术，同时维护合法查询的系统功能。



## **27. TrojanRobot: Physical-world Backdoor Attacks Against VLM-based Robotic Manipulation**

TrojanRobot：针对基于VLM的机器人操纵的物理世界后门攻击 cs.RO

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.11683v5) [paper-pdf](http://arxiv.org/pdf/2411.11683v5)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Aishan Liu, Yunpeng Jiang, Leo Yu Zhang, Xiaohua Jia

**Abstract**: Robotic manipulation in the physical world is increasingly empowered by \textit{large language models} (LLMs) and \textit{vision-language models} (VLMs), leveraging their understanding and perception capabilities. Recently, various attacks against such robotic policies have been proposed, with backdoor attacks drawing considerable attention for their high stealth and strong persistence capabilities. However, existing backdoor efforts are limited to simulators and suffer from physical-world realization. To address this, we propose \textit{TrojanRobot}, a highly stealthy and broadly effective robotic backdoor attack in the physical world. Specifically, we introduce a module-poisoning approach by embedding a backdoor module into the modular robotic policy, enabling backdoor control over the policy's visual perception module thereby backdooring the entire robotic policy. Our vanilla implementation leverages a backdoor-finetuned VLM to serve as the backdoor module. To enhance its generalization in physical environments, we propose a prime implementation, leveraging the LVLM-as-a-backdoor paradigm and developing three types of prime attacks, \ie, \textit{permutation}, \textit{stagnation}, and \textit{intentional} attacks, thus achieving finer-grained backdoors. Extensive experiments on the UR3e manipulator with 18 task instructions using robotic policies based on four VLMs demonstrate the broad effectiveness and physical-world stealth of TrojanRobot. Our attack's video demonstrations are available via a github link https://trojanrobot.github.io.

摘要: \textit{大型语言模型}（LLM）和\textit{视觉语言模型}（VLMS）利用它们的理解和感知能力，越来越多地增强物理世界中的机器人操纵能力。最近，针对此类机器人策略的各种攻击被提出，其中后门攻击因其高隐身性和强持久性能力而引起了相当大的关注。然而，现有的后门工作仅限于模拟器，并且受到物理世界实现的影响。为了解决这个问题，我们提出了\textit{TrojanRobot}，这是物理世界中一种高度隐蔽且广泛有效的机器人后门攻击。具体来说，我们通过将后门模块嵌入模块到模块化机器人策略中来引入模块中毒方法，从而对策略的视觉感知模块进行后门控制，从而后门化整个机器人策略。我们的普通实现利用一个经过后门微调的VLM作为后门模块。为了增强其在物理环境中的通用性，我们提出了一种主要实现，利用LVLM作为后门范式并开发三种类型的主要攻击，即\textit{perspective}、\textit{staduction}和\textit{intentional}攻击，从而实现更细粒度的后门。UR 3e机械手与18个任务指令使用机器人策略的基础上，四个VLMs的广泛实验证明了广泛的有效性和物理世界的隐身TrojanRobot。我们的攻击视频演示可以通过github链接https://trojanrobot.github.io获得。



## **28. Context-Aware Membership Inference Attacks against Pre-trained Large Language Models**

针对预训练大型语言模型的上下文感知成员推断攻击 cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2409.13745v2) [paper-pdf](http://arxiv.org/pdf/2409.13745v2)

**Authors**: Hongyan Chang, Ali Shahin Shamsabadi, Kleomenis Katevas, Hamed Haddadi, Reza Shokri

**Abstract**: Membership Inference Attacks (MIAs) on pre-trained Large Language Models (LLMs) aim at determining if a data point was part of the model's training set. Prior MIAs that are built for classification models fail at LLMs, due to ignoring the generative nature of LLMs across token sequences. In this paper, we present a novel attack on pre-trained LLMs that adapts MIA statistical tests to the perplexity dynamics of subsequences within a data point. Our method significantly outperforms prior approaches, revealing context-dependent memorization patterns in pre-trained LLMs.

摘要: 对预训练的大型语言模型（LLM）的成员推断攻击（MIA）旨在确定数据点是否是模型训练集的一部分。先前为分类模型构建的MIA在LLM上失败，因为忽视了令牌序列之间的LLM的生成性质。在本文中，我们提出了一种对预训练的LLM的新型攻击，该攻击将MIA统计测试适应数据点内子序列的困惑动态。我们的方法显着优于之前的方法，揭示了预训练的LLM中依赖上下文的记忆模式。



## **29. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

大型多模式模型的鲁棒适应用于检索增强仇恨模因检测 cs.CL

EMNLP 2025 Main (Oral)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2502.13061v4) [paper-pdf](http://arxiv.org/pdf/2502.13061v4)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL

摘要: 仇恨模因已成为互联网上的一个重要问题，需要强大的自动化检测系统。虽然大型多模式模型（LSYS）在仇恨模因检测方面表现出了希望，但它们面临着显着的挑战，例如次优的性能和有限的域外概括能力。最近的研究进一步揭示了在这种环境下将监督微调（SFT）和上下文学习应用于LSYS时的局限性。为了解决这些问题，我们提出了一个用于仇恨模因检测的鲁棒适应框架，该框架可以增强领域内准确性和跨领域概括性，同时保留Letts的一般视觉语言能力。分析表明，与SFT模型相比，我们的方法在对抗攻击下实现了更好的鲁棒性。对六个模因分类数据集的实验表明，我们的方法实现了最先进的性能，优于更大的代理系统。此外，与标准SFT相比，我们的方法为解释仇恨内容生成了更高质量的理由，增强了模型的可解释性。代码可访问https://github.com/JingbiaoMei/RGCL



## **30. Beyond Data Privacy: New Privacy Risks for Large Language Models**

超越数据隐私：大型语言模型的新隐私风险 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.14278v1) [paper-pdf](http://arxiv.org/pdf/2509.14278v1)

**Authors**: Yuntao Du, Zitao Li, Ninghui Li, Bolin Ding

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress in natural language understanding, reasoning, and autonomous decision-making. However, these advancements have also come with significant privacy concerns. While significant research has focused on mitigating the data privacy risks of LLMs during various stages of model training, less attention has been paid to new threats emerging from their deployment. The integration of LLMs into widely used applications and the weaponization of their autonomous abilities have created new privacy vulnerabilities. These vulnerabilities provide opportunities for both inadvertent data leakage and malicious exfiltration from LLM-powered systems. Additionally, adversaries can exploit these systems to launch sophisticated, large-scale privacy attacks, threatening not only individual privacy but also financial security and societal trust. In this paper, we systematically examine these emerging privacy risks of LLMs. We also discuss potential mitigation strategies and call for the research community to broaden its focus beyond data privacy risks, developing new defenses to address the evolving threats posed by increasingly powerful LLMs and LLM-powered systems.

摘要: 大型语言模型（LLM）在自然语言理解、推理和自主决策方面取得了显着进展。然而，这些进步也伴随着严重的隐私问题。虽然大量研究的重点是减轻LLM在模型训练的各个阶段的数据隐私风险，但人们对它们部署中出现的新威胁的关注较少。LLM集成到广泛使用的应用程序中以及其自主能力的武器化产生了新的隐私漏洞。这些漏洞为LLM支持的系统无意中泄露数据和恶意泄露提供了机会。此外，对手可以利用这些系统发起复杂的大规模隐私攻击，不仅威胁个人隐私，还威胁金融安全和社会信任。在本文中，我们系统地研究了LLC这些新出现的隐私风险。我们还讨论了潜在的缓解策略，并呼吁研究界将重点扩大到数据隐私风险之外，开发新的防御措施来应对日益强大的LLM和LLM驱动的系统所构成的不断变化的威胁。



## **31. Adversarial Prompt Distillation for Vision-Language Models**

视觉语言模型的对抗性提示蒸馏 cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.15244v3) [paper-pdf](http://arxiv.org/pdf/2411.15244v3)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.

摘要: 对比图像预训练（CLIP）等大型预训练视觉语言模型（VLM）已被证明容易受到对抗攻击，这引发了人们对其在自动驾驶和医疗诊断等安全关键应用中部署的担忧。对抗性提示调整（APT）是对预训练的VLM进行鲁棒化的一种有希望的方法，它在提示调整的过程中应用对抗性训练。然而，现有的APT方法大多是单模式方法，仅为视觉或文本模式设计提示，从而限制了其稳健性或清晰准确性的有效性。在这项工作中，我们提出了对抗性提示蒸馏（APT），这是一个双峰知识蒸馏框架，通过将APT与多模式知识转移集成来增强APT。APT优化视觉和文本模式的提示，同时从干净的预培训教师CLIP模型中提取知识。对多个基准数据集的广泛实验证明了我们的APT方法在对抗稳健性和精确性方面优于当前最先进的APT方法。APT的有效性也验证了使用非稳健教师来提高微调后的VLM的通用性和稳健性的可能性。



## **32. Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection**

视觉上下文攻击：利用图像驱动上下文注入越狱MLLM cs.CV

Accepted to EMNLP 2025 (Main). 17 pages, 7 figures

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2507.02844v2) [paper-pdf](http://arxiv.org/pdf/2507.02844v2)

**Authors**: Ziqi Miao, Yi Ding, Lijun Li, Jing Shao

**Abstract**: With the emergence of strong vision language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: vision-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct vision-focused strategies, dynamically generating auxiliary images when necessary to construct a vision-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which achieves a toxicity score of 2.48 and an ASR of 22.2%. Code: https://github.com/Dtc7w3PQ/Visco-Attack.

摘要: 随着强大视觉语言能力的出现，多模式大型语言模型（MLLM）在现实世界应用中展示了巨大的潜力。然而，视觉模式所表现出的安全漏洞对在开放世界环境中部署此类模型构成了重大挑战。最近的研究通过将有害的文本语义直接编码到视觉输入中，成功地诱导了目标MLLM的有害反应。然而，在这些方法中，视觉形态主要充当不安全行为的触发器，通常表现出语义模糊性并且在现实场景中缺乏基础。在这项工作中，我们定义了一种新颖的环境：以视觉为中心的越狱，其中视觉信息是构建完整而现实的越狱背景的必要组成部分。在此设置的基础上，我们提出了VisCo（视觉上下文）攻击。VisCo使用四种不同的以视觉为中心的策略构建上下文对话，在必要时动态生成辅助图像，以构建以视觉为中心的越狱场景。为了最大限度地提高攻击效果，它结合了自动毒性混淆和语义细化，以产生最终的攻击提示，从而可靠地触发目标黑匣子MLLM的有害响应。具体而言，VisCo在MM-SafetyBench上对GPT-4 o的毒性评分为4.78，攻击成功率（ASB）为85%，显着优于基线，基线达到了2.48的毒性评分和22.2%的ASB。代码：https://github.com/Dtc7w3PQ/Visco-Attack。



## **33. Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content**

迈向包容性有毒内容适度：解决毒性分类器中对抗性攻击的漏洞 cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12672v1) [paper-pdf](http://arxiv.org/pdf/2509.12672v1)

**Authors**: Shaz Furniturewala, Arkaitz Zubiaga

**Abstract**: The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.

摘要: 由于大型语言模型（LLM）的广泛使用，在线机器生成内容的数量急剧增长，这给内容审核系统带来了新的挑战。传统的内容审核分类器通常根据人类生成的文本进行训练，但由于LLM生成的文本偏离其训练数据以及旨在避免检测的对抗性攻击而遭受错误分类。当今的防御策略是被动的，而不是主动的，因为它们依赖于对抗训练或外部检测模型来识别攻击。在这项工作中，我们的目标是识别毒性分类器中导致错误分类的脆弱组件，提出一种基于机械解释性技术的新型策略。我们的研究重点是微调的BERT和RoBERTa分类器，对跨越各种少数群体的不同数据集进行测试。我们使用对抗攻击技术来识别脆弱的电路。最后，我们抑制了这些脆弱的电路，提高了对抗攻击的性能。我们还提供了对这些脆弱电路的人口统计学层面的见解，揭示了模型训练中的公平性和稳健性差距。我们发现模型具有不同的头部，这些头部要么对性能至关重要，要么容易受到攻击，而抑制脆弱的头部可以提高对抗性输入的性能。我们还发现，不同的头部导致了不同人口群体的脆弱性，这可以为毒性检测模型的更具包容性的开发提供信息。



## **34. A Systematic Evaluation of Parameter-Efficient Fine-Tuning Methods for the Security of Code LLMs**

代码LLM安全性的参数高效微调方法的系统评估 cs.CR

25 pages

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12649v1) [paper-pdf](http://arxiv.org/pdf/2509.12649v1)

**Authors**: Kiho Lee, Jungkon Kim, Doowon Kim, Hyoungshick Kim

**Abstract**: Code-generating Large Language Models (LLMs) significantly accelerate software development. However, their frequent generation of insecure code presents serious risks. We present a comprehensive evaluation of seven parameter-efficient fine-tuning (PEFT) techniques, demonstrating substantial gains in secure code generation without compromising functionality. Our research identifies prompt-tuning as the most effective PEFT method, achieving an 80.86% Overall-Secure-Rate on CodeGen2 16B, a 13.5-point improvement over the 67.28% baseline. Optimizing decoding strategies through sampling temperature further elevated security to 87.65%. This equates to a reduction of approximately 203,700 vulnerable code snippets per million generated. Moreover, prompt and prefix tuning increase robustness against poisoning attacks in our TrojanPuzzle evaluation, with strong performance against CWE-79 and CWE-502 attack vectors. Our findings generalize across Python and Java, confirming prompt-tuning's consistent effectiveness. This study provides essential insights and practical guidance for building more resilient software systems with LLMs.

摘要: 代码生成大型语言模型（LLM）显着加速了软件开发。然而，他们频繁生成不安全的代码带来了严重的风险。我们对七种参数高效微调（PEFT）技术进行了全面评估，展示了在不损害功能的情况下在安全代码生成方面的巨大收益。我们的研究将预算调整确定为最有效的PEFT方法，在CodeGen 2 16 B上实现了80.86%的总体安全率，比67.28%的基线提高了13.5个百分点。通过采样温度优化解码策略，安全性进一步提高至87.65%。这相当于每生成的百万个易受攻击的代码片段减少约203，700个。此外，在我们的TrojanPuzzle评估中，提示和前置调整增强了针对中毒攻击的鲁棒性，在针对CWE-79和CWE-502攻击载体的性能强劲。我们的研究结果在Python和Java中得到了推广，证实了预算调优的一致有效性。这项研究为使用LLM构建更具弹性的软件系统提供了重要的见解和实践指导。



## **35. Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs**

LLM联合量化和稀疏化的最佳大脑恢复 cs.CL

Preprint

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11177v2) [paper-pdf](http://arxiv.org/pdf/2509.11177v2)

**Authors**: Hang Guo, Yawei Li, Luca Benini

**Abstract**: Recent advances in Large Language Model (LLM) compression, such as quantization and pruning, have achieved notable success. However, as these techniques gradually approach their respective limits, relying on a single method for further compression has become increasingly challenging. In this work, we explore an alternative solution by combining quantization and sparsity. This joint approach, though promising, introduces new difficulties due to the inherently conflicting requirements on weight distributions: quantization favors compact ranges, while pruning benefits from high variance. To attack this problem, we propose Optimal Brain Restoration (OBR), a general and training-free framework that aligns pruning and quantization by error compensation between both. OBR minimizes performance degradation on downstream tasks by building on a second-order Hessian objective, which is then reformulated into a tractable problem through surrogate approximation and ultimately reaches a closed-form solution via group error compensation. Experiments show that OBR enables aggressive W4A4KV4 quantization with 50% sparsity on existing LLMs, and delivers up to 4.72x speedup and 6.4x memory reduction compared to the FP16-dense baseline.

摘要: 大型语言模型（LLM）压缩的最新进展（例如量化和修剪）取得了显着的成功。然而，随着这些技术逐渐接近各自的极限，依靠单一方法进行进一步压缩变得越来越具有挑战性。在这项工作中，我们通过结合量化和稀疏性来探索替代解决方案。这种联合方法虽然很有希望，但由于对权重分布的固有要求相互冲突，带来了新的困难：量化有利于紧凑的范围，而修剪则受益于高方差。为了解决这个问题，我们提出了最佳大脑恢复（OBR），这是一个通用的、免训练的框架，通过两者之间的误差补偿来协调修剪和量化。OBR通过建立二阶Hessian目标来最大限度地减少下游任务的性能下降，然后通过代理逼近将该目标重新表述为可处理的问题，并最终通过群误差补偿获得封闭形式的解决方案。实验表明，OBR可以在现有LLM上以50%的稀疏度实现积极的W4 A4 KV 4量化，与FP 16密集基线相比，可提供高达4.72倍的加速和6.4倍的内存减少。



## **36. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

EmantSleuth：通过语义意图不变性检测提示注入 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.20890v2) [paper-pdf](http://arxiv.org/pdf/2508.20890v2)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.

摘要: 大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，从虚拟助手到自治代理。然而，它们的灵活性也引入了新的攻击向量，特别是提示注入（PI），其中攻击者通过精心制作的输入操纵模型行为。随着攻击者不断地使用释义、混淆甚至多任务注入策略，现有的基准不再足以捕获所有新兴威胁。   为了解决这一差距，我们构建了一个新的基准，系统地扩展了以前的努力。我们的基准涵盖了两种广泛使用的现有基准，同时引入了新的操纵技术和多任务场景，从而提供了更全面的评估设置。我们发现，现有的防御虽然在原始基准上有效，但在我们的基准下表现出明显的弱点，这凸显了对更强大解决方案的需求。我们的关键见解是，虽然攻击形式可能会有所不同，但对手的意图（注入未经授权的任务）保持不变。在这一观察的基础上，我们提出了EmittSleuth，这是一个面向语义的防御框架，它通过对任务级意图而不是表面特征进行推理来检测提示注入。在最先进的基准测试中进行评估后，AktSleuth始终优于现有的防御，同时保持相当的运行时间和成本效率。这些结果表明，基于意图的语义推理提供了一个强大的，有效的，和可推广的策略，以抵御不断发展的即时注入威胁的LLM。



## **37. Phi: Preference Hijacking in Multi-modal Large Language Models at Inference Time**

Phi：推理时多模式大型语言模型中的偏好劫持 cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.12521v1) [paper-pdf](http://arxiv.org/pdf/2509.12521v1)

**Authors**: Yifan Lan, Yuanpu Cao, Weitong Zhang, Lu Lin, Jinghui Chen

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have gained significant attention across various domains. However, their widespread adoption has also raised serious safety concerns. In this paper, we uncover a new safety risk of MLLMs: the output preference of MLLMs can be arbitrarily manipulated by carefully optimized images. Such attacks often generate contextually relevant yet biased responses that are neither overtly harmful nor unethical, making them difficult to detect. Specifically, we introduce a novel method, Preference Hijacking (Phi), for manipulating the MLLM response preferences using a preference hijacked image. Our method works at inference time and requires no model modifications. Additionally, we introduce a universal hijacking perturbation -- a transferable component that can be embedded into different images to hijack MLLM responses toward any attacker-specified preferences. Experimental results across various tasks demonstrate the effectiveness of our approach. The code for Phi is accessible at https://github.com/Yifan-Lan/Phi.

摘要: 最近，多模式大型语言模型（MLLM）在各个领域引起了密切关注。然而，它们的广泛采用也引发了严重的安全问题。在本文中，我们发现了MLLM的一个新的安全风险：MLLM的输出偏好可以通过精心优化的图像任意操纵。此类攻击通常会产生与上下文相关但有偏见的反应，既不明显有害，也不不道德，因此难以检测。具体来说，我们引入了一种新颖的方法，即偏好劫持（Phi），用于使用偏好劫持的图像来操纵MLLM响应偏好。我们的方法在推理时工作，不需要模型修改。此外，我们还引入了一种通用的劫持扰动--一种可移植的组件，可以嵌入到不同的图像中，以劫持MLLM对任何攻击者指定的偏好的响应。各种任务的实验结果证明了我们方法的有效性。Phi的代码可在https://github.com/Yifan-Lan/Phi上访问。



## **38. Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering**

保持安全！针对问题解答中的间接攻击，对大型语言模型上下文中的安全策略保留进行基准测试 cs.CL

EMNLP 2025 (Main Conference)

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2505.15805v2) [paper-pdf](http://arxiv.org/pdf/2505.15805v2)

**Authors**: Hwan Chang, Yumin Kim, Yonghyun Jun, Hwanhee Lee

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains such as enterprise and government, ensuring that they adhere to user-defined security policies within context is critical-especially with respect to information non-disclosure. While prior LLM studies have focused on general safety and socially sensitive data, large-scale benchmarks for contextual security preservation against attacks remain lacking. To address this, we introduce a novel large-scale benchmark dataset, CoPriva, evaluating LLM adherence to contextual non-disclosure policies in question answering. Derived from realistic contexts, our dataset includes explicit policies and queries designed as direct and challenging indirect attacks seeking prohibited information. We evaluate 10 LLMs on our benchmark and reveal a significant vulnerability: many models violate user-defined policies and leak sensitive information. This failure is particularly severe against indirect attacks, highlighting a critical gap in current LLM safety alignment for sensitive applications. Our analysis reveals that while models can often identify the correct answer to a query, they struggle to incorporate policy constraints during generation. In contrast, they exhibit a partial ability to revise outputs when explicitly prompted. Our findings underscore the urgent need for more robust methods to guarantee contextual security.

摘要: 随着大型语言模型（LLM）越来越多地部署在企业和政府等敏感领域，确保它们在上下文中遵守用户定义的安全策略至关重要，尤其是在信息不披露方面。虽然之前的LLM研究重点关注一般安全和社会敏感数据，但仍然缺乏针对攻击的上下文安全保护的大规模基准。为了解决这个问题，我们引入了一个新颖的大规模基准数据集CoPriva，以评估LLM在问答中对上下文保密政策的遵守情况。我们的数据集源自现实背景，包括明确的政策和查询，旨在作为寻求违禁信息的直接和具有挑战性的间接攻击。我们在我们的基准测试中评估了10个LLM，并揭示了一个重大漏洞：许多模型违反了用户定义的策略并泄漏了敏感信息。这种故障对于间接攻击尤其严重，突出了当前LLM安全对齐敏感应用程序的关键差距。我们的分析表明，虽然模型通常可以识别查询的正确答案，但它们很难在生成过程中纳入政策约束。相比之下，他们表现出部分的能力，修改输出时，明确提示。我们的研究结果强调迫切需要更强大的方法来保证上下文安全。



## **39. Early Approaches to Adversarial Fine-Tuning for Prompt Injection Defense: A 2022 Study of GPT-3 and Contemporary Models**

即时注射防御对抗微调的早期方法：2022年GPT-3和当代模型的研究 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.14271v1) [paper-pdf](http://arxiv.org/pdf/2509.14271v1)

**Authors**: Gustavo Sandoval, Denys Fenchenko, Junyao Chen

**Abstract**: This paper documents early research conducted in 2022 on defending against prompt injection attacks in large language models, providing historical context for the evolution of this critical security domain. This research focuses on two adversarial attacks against Large Language Models (LLMs): prompt injection and goal hijacking. We examine how to construct these attacks, test them on various LLMs, and compare their effectiveness. We propose and evaluate a novel defense technique called Adversarial Fine-Tuning. Our results show that, without this defense, the attacks succeeded 31\% of the time on GPT-3 series models. When using our Adversarial Fine-Tuning approach, attack success rates were reduced to near zero for smaller GPT-3 variants (Ada, Babbage, Curie), though we note that subsequent research has revealed limitations of fine-tuning-based defenses. We also find that more flexible models exhibit greater vulnerability to these attacks. Consequently, large models such as GPT-3 Davinci are more vulnerable than smaller models like GPT-2. While the specific models tested are now superseded, the core methodology and empirical findings contributed to the foundation of modern prompt injection defense research, including instruction hierarchy systems and constitutional AI approaches.

摘要: 本文记录了2022年针对大型语言模型中的即时注入攻击进行的早期研究，为这一关键安全领域的演变提供了历史背景。本研究重点关注针对大型语言模型（LLM）的两种对抗攻击：提示注入和目标劫持。我们研究如何构建这些攻击，在各种LLM上测试它们，并比较它们的有效性。我们提出并评估了一种名为对抗微调的新型防御技术。我们的结果表明，如果没有这种防御，GPT-3系列模型上的攻击成功率为31%。当使用我们的对抗性微调方法时，较小的GPT-3变体（Ada、Babbage、Curie）的攻击成功率降至接近零，尽管我们注意到后续研究揭示了基于微调的防御的局限性。我们还发现，更灵活的模型对这些攻击表现出更大的脆弱性。因此，GPT-3 Davinci等大型型号比GPT-2等小型型号更容易受到攻击。虽然测试的具体模型现在已被取代，但核心方法论和经验发现为现代即时注射防御研究的基础做出了贡献，包括指令层次系统和宪法人工智能方法。



## **40. Safety Pretraining: Toward the Next Generation of Safe AI**

安全预培训：迈向下一代安全人工智能 cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2504.16980v2) [paper-pdf](http://arxiv.org/pdf/2504.16980v2)

**Authors**: Pratyush Maini, Sachin Goyal, Dylan Sam, Alex Robey, Yash Savani, Yiding Jiang, Andy Zou, Matt Fredrikson, Zacharcy C. Lipton, J. Zico Kolter

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes settings, the risk of generating harmful or toxic content remains a central challenge. Post-hoc alignment methods are brittle: once unsafe patterns are learned during pretraining, they are hard to remove. In this work, we present a data-centric pretraining framework that builds safety into the model from the start. Our framework consists of four key steps: (i) Safety Filtering: building a safety classifier to classify webdata into safe and unsafe categories; (ii) Safety Rephrasing: we recontextualize unsafe webdata into safer narratives; (iii) Native Refusal: we develop RefuseWeb and Moral Education pretraining datasets that actively teach model to refuse on unsafe content and the moral reasoning behind it, and (iv) Harmfulness-Tag annotated pretraining: we flag unsafe content during pretraining using a special token, and use it to steer model away from unsafe generations at inference. Our safety-pretrained models reduce attack success rates from 38.8\% to 8.4\% on standard LLM safety benchmarks with no performance degradation on general tasks.

摘要: 随着大型语言模型（LLM）越来越多地部署在高风险环境中，生成有害或有毒内容的风险仍然是一个核心挑战。事后对齐方法很脆弱：一旦在预训练期间学习到不安全的模式，它们就很难被删除。在这项工作中，我们提出了一个以数据为中心的预训练框架，该框架从一开始就将安全性构建到模型中。我们的框架由四个关键步骤组成：（i）安全过滤：构建一个安全分类器，将网络数据分为安全和不安全类别;（ii）安全改写：我们将不安全的网络数据重新语境化为更安全的叙述;（iii）原生拒绝：我们开发RefuseWeb和道德教育预训练数据集，积极教导模型拒绝不安全内容及其背后的道德推理，和（iv）有害标签注释的预训练：我们在预训练期间使用特殊令牌标记不安全内容，并使用它在推理时引导模型远离不安全的世代。我们的安全预训练模型将标准LLM安全基准的攻击成功率从38.8%降低到8.4%，而一般任务的性能不会下降。



## **41. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

NeuronStrike：对对齐的LLM的神经元级攻击 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11864v1) [paper-pdf](http://arxiv.org/pdf/2509.11864v1)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.

摘要: 安全一致对于大型语言模型（LLM）的道德部署至关重要，指导它们避免生成有害或不道德的内容。当前的对齐技术，例如有监督的微调和来自人类反馈的强化学习，仍然很脆弱，可以被精心设计的对抗提示绕过。不幸的是，此类攻击依赖于试错，缺乏跨模型的通用性，并且受到可扩展性和可靠性的限制。   本文介绍了NeuroStrike，这是一种新颖且可推广的攻击框架，它利用了对齐技术引入的一个基本漏洞：依赖于负责检测和抑制有害输入的稀疏、专门的安全神经元。我们将NeuroStrike应用于白盒和黑盒设置：在白盒设置中，NeuroStrike通过反馈激活分析识别安全神经元，并在推理期间修剪它们以禁用安全机制。在黑匣子环境中，我们提出了第一次LLM剖析攻击，该攻击通过在开权重代理模型上训练对抗提示生成器，然后将它们部署到黑匣子和专有目标上来利用安全神经元的可移植性。我们对来自主要LLM开发商的20多个开量级LLM进行了评估。通过删除目标层中不到0.6%的神经元，NeuroStrike仅使用普通恶意提示即可实现76.9%的平均攻击成功率（ASB）。此外，Neurostrike将四种多模式LLM推广到对不安全图像输入具有100%的ASB。安全神经元在架构之间有效转移，使11个微调模型的ASB达到78.5%，5个提炼模型的ASB达到77.7%。黑匣子LLM分析攻击在包括Google Gemini系列在内的五种黑匣子型号中实现了63.7%的平均ASB。



## **42. One Goal, Many Challenges: Robust Preference Optimization Amid Content-Aware and Multi-Source Noise**

一个目标，诸多挑战：内容感知和多源噪音中的稳健偏好优化 cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2503.12301v2) [paper-pdf](http://arxiv.org/pdf/2503.12301v2)

**Authors**: Amirabbas Afzali, Amirhossein Afsharrad, Seyed Shahabeddin Mousavi, Sanjay Lall

**Abstract**: Large Language Models (LLMs) have made significant strides in generating human-like responses, largely due to preference alignment techniques. However, these methods often assume unbiased human feedback, which is rarely the case in real-world scenarios. This paper introduces Content-Aware Noise-Resilient Preference Optimization (CNRPO), a novel framework that addresses multiple sources of content-dependent noise in preference learning. CNRPO employs a multi-objective optimization approach to separate true preferences from content-aware noises, effectively mitigating their impact. We leverage backdoor attack mechanisms to efficiently learn and control various noise sources within a single model. Theoretical analysis and extensive experiments on different synthetic noisy datasets demonstrate that CNRPO significantly improves alignment with primary human preferences while controlling for secondary noises and biases, such as response length and harmfulness.

摘要: 大型语言模型（LLM）在生成类人响应方面取得了重大进展，这主要归功于偏好对齐技术。然而，这些方法通常假设无偏见的人类反馈，而在现实世界场景中情况很少。本文介绍了内容感知噪音弹性偏好优化（CNRPO），这是一种新型框架，可解决偏好学习中多个内容相关噪音来源。CNRPO采用多目标优化方法将真实偏好与内容感知噪音分开，有效减轻其影响。我们利用后门攻击机制来有效地学习和控制单个模型内的各种噪音源。对不同合成噪音数据集的理论分析和广泛实验表明，CNRPO显着改善了与人类主要偏好的一致性，同时控制次要噪音和偏差，例如响应长度和危害性。



## **43. Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check**

合理的安全调整：通过先检查确保越狱防御 cs.LG

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11629v1) [paper-pdf](http://arxiv.org/pdf/2509.11629v1)

**Authors**: Chentao Cao, Xiaojun Xu, Bo Han, Hang Li

**Abstract**: As large language models (LLMs) continue to advance in capabilities, ensuring their safety against jailbreak attacks remains a critical challenge. In this paper, we introduce a novel safety alignment approach called Answer-Then-Check, which enhances LLM robustness against malicious prompts by applying thinking ability to mitigate jailbreaking problems before producing a final answer to the user. Our method enables models to directly answer the question in their thought and then critically evaluate its safety before deciding whether to provide it. To implement this approach, we construct the Reasoned Safety Alignment (ReSA) dataset, comprising 80K examples that teach models to reason through direct responses and then analyze their safety. Experimental results demonstrate that our approach achieves the Pareto frontier with superior safety capability while decreasing over-refusal rates on over-refusal benchmarks. Notably, the model fine-tuned with ReSA maintains general reasoning capabilities on benchmarks like MMLU, MATH500, and HumanEval. Besides, our method equips models with the ability to perform safe completion. Unlike post-hoc methods that can only reject harmful queries, our model can provide helpful and safe alternative responses for sensitive topics (e.g., self-harm). Furthermore, we discover that training on a small subset of just 500 examples can achieve comparable performance to using the full dataset, suggesting that safety alignment may require less data than previously assumed.

摘要: 随着大型语言模型（LLM）的能力不断进步，确保其免受越狱攻击的安全性仍然是一个严峻的挑战。在本文中，我们引入了一种名为“Searcher-Then-Check”的新型安全对齐方法，该方法通过在向用户生成最终答案之前应用思维能力来缓解越狱问题，来增强LLM针对恶意提示的鲁棒性。我们的方法使模型能够在思想中直接回答问题，然后在决定是否提供之前批判性地评估其安全性。为了实施这种方法，我们构建了推理安全对齐（ReSA）数据集，其中包括8万个示例，教模型通过直接响应进行推理，然后分析其安全性。实验结果表明，我们的方法以卓越的安全能力达到了帕累托前沿，同时降低了过度拒绝基准上的过度拒绝率。值得注意的是，使用ReSA微调的模型在MMLU，MATH 500和HumanEval等基准上保持了一般推理能力。此外，我们的方法装备模型的能力，执行安全完成。与只能拒绝有害查询的事后方法不同，我们的模型可以为敏感主题（例如，自我伤害）。此外，我们发现，仅对500个示例的一小部分进行训练就可以获得与使用完整数据集相当的性能，这表明安全性对齐可能需要比之前假设的更少的数据。



## **44. Multilingual Collaborative Defense for Large Language Models**

大型语言模型的多语言协作防御 cs.CL

21 pages, 4figures

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2505.11835v2) [paper-pdf](http://arxiv.org/pdf/2505.11835v2)

**Authors**: Hongliang Li, Jinan Xu, Gengping Cui, Changhao Guan, Fengran Mo, Kaiyu Huang

**Abstract**: The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at https://github.com/HLiang-Lee/MCD.

摘要: 大型语言模型（LLM）的稳健性和安全性已成为一个重要的研究领域。一个值得注意的漏洞是，通过将有害查询翻译成罕见或代表性不足的语言来绕过LLM保障措施，这是“越狱”这些模型的一种简单而有效的方法。尽管人们的担忧日益加剧，但针对多语言场景下LLM保护的研究有限，凸显了加强多语言安全的迫切需要。在这项工作中，我们调查了不同语言中的各种攻击特征之间的相关性，并提出了多语言协作防御（MCB），这是一种新型学习方法，可以自动优化连续的软安全提示，以促进LLM的多语言保护。MCB方法具有三个优点：首先，它有效地提高了跨多种语言的性能保护。其次，MCB保持了强大的概括能力，同时最大限度地降低了错误拒绝率。第三，MCB缓解了LLM培训库失衡造成的语言安全失调。为了评估MCB的有效性，我们手动构建常用越狱基准（例如MaliciousDirecct和AdvBench）的多语言版本，以评估各种保障方法。此外，我们以未充分代表（零镜头）语言引入这些数据集，以验证MCB的语言可移植性。结果表明，MCB在防范多语言越狱企图方面优于现有方法，同时还表现出强大的语言传输能力。我们的代码可在https://github.com/HLiang-Lee/MCD上获取。



## **45. Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs**

混乱是最后的障碍：重新思考越狱评估并调查LLM的真正滥用威胁 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2508.16347v2) [paper-pdf](http://arxiv.org/pdf/2508.16347v2)

**Authors**: Yu Yan, Sheng Sun, Zhe Wang, Yijun Lin, Zenghao Duan, zhifei zheng, Min Liu, Zhiyi yin, Jianping Zhang

**Abstract**: With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential.

摘要: 随着大型语言模型（LLM）的发展，许多努力揭示了它们对越狱攻击的脆弱性。尽管这些研究推动了LLM安全调整的进展，但目前尚不清楚LLM是否已经内化了真实的知识来应对现实世界的犯罪，或者只是被迫模拟有毒的语言模式。这种模糊性引发了人们的担忧，即越狱成功通常归因于越狱LLM和法官LLM之间的幻觉循环。通过脱钩越狱技术的使用，我们构建知识密集型问答，以调查LLM在危险知识拥有、有害任务规划效用和有害判断稳健性方面的滥用威胁。实验揭示了LLM的越狱成功率和有害知识拥有之间的不匹配，而现有的LLM作为法官框架往往会将有害判断锚定在有毒语言模式上。我们的研究揭示了现有LLM安全评估与现实世界潜在威胁之间的差距。



## **46. Enhancing Prompt Injection Attacks to LLMs via Poisoning Alignment**

通过中毒对齐增强对LLM的即时注入攻击 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2410.14827v3) [paper-pdf](http://arxiv.org/pdf/2410.14827v3)

**Authors**: Zedian Shao, Hongbin Liu, Jaden Mu, Neil Zhenqiang Gong

**Abstract**: Prompt injection attack, where an attacker injects a prompt into the original one, aiming to make an Large Language Model (LLM) follow the injected prompt to perform an attacker-chosen task, represent a critical security threat. Existing attacks primarily focus on crafting these injections at inference time, treating the LLM itself as a static target. Our experiments show that these attacks achieve some success, but there is still significant room for improvement. In this work, we introduces a more foundational attack vector: poisoning the LLM's alignment process to amplify the success of future prompt injection attacks. Specifically, we propose PoisonedAlign, a method that strategically creates poisoned alignment samples to poison an LLM's alignment dataset. Our experiments across five LLMs and two alignment datasets show that when even a small fraction of the alignment data is poisoned, the resulting model becomes substantially more vulnerable to a wide range of prompt injection attacks. Crucially, this vulnerability is instilled while the LLM's performance on standard capability benchmarks remains largely unchanged, making the manipulation difficult to detect through automated, general-purpose performance evaluations. The code for implementing the attack is available at https://github.com/Sadcardation/PoisonedAlign.

摘要: 提示注入攻击是指攻击者将提示注入到原始提示中，旨在使大型语言模型（LLM）遵循注入的提示来执行攻击者选择的任务，这代表了一种严重的安全威胁。现有的攻击主要集中在推理时制作这些注入，将LLM本身视为静态目标。我们的实验表明，这些攻击取得了一定成功，但仍有很大的改进空间。在这项工作中，我们引入了一个更基本的攻击载体：毒害LLM的对齐过程，以放大未来即时注入攻击的成功。具体来说，我们提出了PoisonedAlign，这是一种策略性地创建有毒比对样本以毒害LLM的比对数据集的方法。我们在五个LLM和两个对齐数据集上进行的实验表明，当即使是一小部分对齐数据被毒害时，生成的模型也会变得更容易受到广泛的即时注入攻击。至关重要的是，该漏洞是在LLM在标准能力基准上的性能基本保持不变的情况下灌输的，从而使得操纵很难通过自动化的通用性能评估检测到。实施攻击的代码可在https://github.com/Sadcardation/PoisonedAlign上获取。



## **47. Securing AI Agents: Implementing Role-Based Access Control for Industrial Applications**

保护人工智能代理：为工业应用程序实施基于角色的访问控制 cs.AI

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.11431v1) [paper-pdf](http://arxiv.org/pdf/2509.11431v1)

**Authors**: Aadil Gani Ganie

**Abstract**: The emergence of Large Language Models (LLMs) has significantly advanced solutions across various domains, from political science to software development. However, these models are constrained by their training data, which is static and limited to information available up to a specific date. Additionally, their generalized nature often necessitates fine-tuning -- whether for classification or instructional purposes -- to effectively perform specific downstream tasks. AI agents, leveraging LLMs as their core, mitigate some of these limitations by accessing external tools and real-time data, enabling applications such as live weather reporting and data analysis. In industrial settings, AI agents are transforming operations by enhancing decision-making, predictive maintenance, and process optimization. For example, in manufacturing, AI agents enable near-autonomous systems that boost productivity and support real-time decision-making. Despite these advancements, AI agents remain vulnerable to security threats, including prompt injection attacks, which pose significant risks to their integrity and reliability. To address these challenges, this paper proposes a framework for integrating Role-Based Access Control (RBAC) into AI agents, providing a robust security guardrail. This framework aims to support the effective and scalable deployment of AI agents, with a focus on on-premises implementations.

摘要: 大型语言模型（LLM）的出现为从政治科学到软件开发的各个领域带来了显着的先进解决方案。然而，这些模型受到其训练数据的限制，训练数据是静态的，并且仅限于特定日期之前可用的信息。此外，它们的普遍性通常需要进行微调（无论是出于分类还是教学目的），以有效地执行特定的下游任务。人工智能代理以LLM为核心，通过访问外部工具和实时数据来缓解其中一些限制，从而启用实时天气报告和数据分析等应用程序。在工业环境中，人工智能代理正在通过加强决策、预测性维护和流程优化来改变运营。例如，在制造业中，人工智能代理支持近乎自主的系统，从而提高生产力并支持实时决策。尽管取得了这些进步，人工智能代理仍然容易受到安全威胁，包括即时注入攻击，这对其完整性和可靠性构成了重大风险。为了应对这些挑战，本文提出了一个将基于角色的访问控制（RSC）集成到人工智能代理中的框架，以提供强大的安全护栏。该框架旨在支持人工智能代理的有效且可扩展的部署，重点关注本地实施。



## **48. Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs**

以毒攻毒（F3）：LVLM中一种无需培训且高效的视觉对抗示例净化方法 cs.CV

Accepted by ACM Multimedia 2025 BNI track (Oral)

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.01064v3) [paper-pdf](http://arxiv.org/pdf/2506.01064v3)

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive ``fighting fire with fire'' strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code is available at https://github.com/btzyd/F3.

摘要: 大型视觉语言模型（LVLM）的最新进展展示了它们在广泛的多模式视觉语言任务中的非凡能力。然而，这些模型仍然容易受到视觉对抗攻击，这可能会极大地损害其性能。在本文中，我们介绍了F3，这是一种新型的对抗净化框架，它采用了违反直觉的“以毒攻毒”策略：有意地向对抗性示例引入简单的扰动以减轻其有害影响。具体来说，F3利用从随机干扰的对手示例中获得的跨模式注意力作为参考目标。通过向这些对抗性示例中注入噪音，F3有效地细化了他们的注意力，从而产生更干净、更可靠的模型输出。值得注意的是，这种看似矛盾的利用噪音来抵消对抗攻击的方法产生了令人印象深刻的净化结果。此外，F3具有几个明显的优势：无需训练且易于实施，并且与现有的纯化方法相比，计算效率显着提高。这些属性使F3特别适合大规模工业应用，其中稳健的性能和运营效率都是关键优先事项。该代码可在https://github.com/btzyd/F3上获取。



## **49. Beyond the Protocol: Unveiling Attack Vectors in the Model Context Protocol (MCP) Ecosystem**

超越协议：揭开模型上下文协议（HCP）生态系统中的攻击载体 cs.CR

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2506.02040v4) [paper-pdf](http://arxiv.org/pdf/2506.02040v4)

**Authors**: Hao Song, Yiming Shen, Wenxuan Luo, Leixin Guo, Ting Chen, Jiashui Wang, Beibei Li, Xiaosong Zhang, Jiachi Chen

**Abstract**: The Model Context Protocol (MCP) is an emerging standard designed to enable seamless interaction between Large Language Model (LLM) applications and external tools or resources. Within a short period, thousands of MCP services have been developed and deployed. However, the client-server integration architecture inherent in MCP may expand the attack surface against LLM Agent systems, introducing new vulnerabilities that allow attackers to exploit by designing malicious MCP servers. In this paper, we present the first end-to-end empirical evaluation of attack vectors targeting the MCP ecosystem. We identify four categories of attacks, i.e., Tool Poisoning Attacks, Puppet Attacks, Rug Pull Attacks, and Exploitation via Malicious External Resources. To evaluate their feasibility, we conduct experiments following the typical steps of launching an attack through malicious MCP servers: upload -> download -> attack. Specifically, we first construct malicious MCP servers and successfully upload them to three widely used MCP aggregation platforms. The results indicate that current audit mechanisms are insufficient to identify and prevent these threats. Next, through a user study and interview with 20 participants, we demonstrate that users struggle to identify malicious MCP servers and often unknowingly install them from aggregator platforms. Finally, we empirically demonstrate that these attacks can trigger harmful actions within the user's local environment, such as accessing private files or controlling devices to transfer digital assets. Additionally, based on interview results, we discuss four key challenges faced by the current MCP security ecosystem. These findings underscore the urgent need for robust security mechanisms to defend against malicious MCP servers and ensure the safe deployment of increasingly autonomous LLM agents.

摘要: 模型上下文协议（HCP）是一种新兴标准，旨在实现大型语言模型（LLM）应用程序与外部工具或资源之间的无缝交互。在短时间内，就开发和部署了数千项HCP服务。然而，LCP固有的客户端-服务器集成架构可能会扩大针对LLM Agent系统的攻击面，引入新的漏洞，允许攻击者通过设计恶意的LCP服务器来利用这些漏洞。本文中，我们首次对针对LCP生态系统的攻击载体进行了端到端的实证评估。我们确定了四类攻击，即工具中毒攻击、木偶攻击、拉地毯攻击以及通过恶意外部资源进行的剥削。为了评估其可行性，我们按照通过恶意LCP服务器发起攻击的典型步骤进行了实验：上传->下载->攻击。具体来说，我们首先构建恶意的LCP服务器，并成功将其上传到三个广泛使用的LCP聚合平台。结果表明，当前的审计机制不足以识别和预防这些威胁。接下来，通过用户研究和对20名参与者的采访，我们证明用户很难识别恶意的LCP服务器，并且通常在不知不觉中从聚合平台安装它们。最后，我们通过经验证明，这些攻击可能会在用户本地环境中引发有害行为，例如访问私人文件或控制设备传输数字资产。此外，根据采访结果，我们讨论了当前HCP安全生态系统面临的四个关键挑战。这些发现凸显了迫切需要强大的安全机制来抵御恶意的HCP服务器，并确保安全部署日益自主的LLM代理。



## **50. Character-Level Perturbations Disrupt LLM Watermarks**

初级扰动扰乱LLM水印 cs.CR

accepted by Network and Distributed System Security (NDSS) Symposium  2026

**SubmitDate**: 2025-09-14    [abs](http://arxiv.org/abs/2509.09112v2) [paper-pdf](http://arxiv.org/pdf/2509.09112v2)

**Authors**: Zhaoxi Zhang, Xiaomei Zhang, Yanjun Zhang, He Zhang, Shirui Pan, Bo Liu, Asif Qumer Gill, Leo Yu Zhang

**Abstract**: Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries.   To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.

摘要: 大型语言模型（LLM）水印将可检测信号嵌入到生成的文本中，以实现版权保护、防止滥用和内容检测。虽然之前的研究使用水印去除攻击来评估稳健性，但这些方法通常不是最优的，从而产生了一种误解，即有效的去除需要大的扰动或强大的对手。   为了弥合差距，我们首先形式化LLM水印的系统模型，并描述了两个受有限访问水印检测器限制的现实威胁模型。然后，我们分析不同类型的扰动在其攻击范围内的变化，即，一次编辑可以影响的代币数量。我们观察到字符级扰动（例如，拼写错误、互换、删除、同字形）可以通过扰乱标记化过程同时影响多个标记。我们证明，在最严格的威胁模型下，字符级扰动对于水印去除来说明显更有效。我们进一步提出了基于遗传算法（GA）的引导删除攻击，使用一个参考检测器进行优化。在一个实际的威胁模型与有限的黑盒查询的水印检测器，我们的方法表现出强大的去除性能。实验证实了字符级扰动的优越性和遗传算法在现实约束条件下去除水印的有效性。此外，我们认为有一个对抗性的困境时，考虑潜在的防御：任何固定的防御可以绕过一个合适的扰动策略。受此原则的启发，我们提出了一种自适应复合字符级攻击。实验结果表明，这种方法可以有效地击败防御。我们的研究结果强调了现有LLM水印方案中的重大漏洞，并强调了开发新的稳健机制的紧迫性。



