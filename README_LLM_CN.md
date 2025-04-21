# Latest Large Language Model Attack Papers
**update at 2025-04-21 09:56:15**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. BadApex: Backdoor Attack Based on Adaptive Optimization Mechanism of Black-box Large Language Models**

BadApex：基于黑匣子大型语言模型自适应优化机制的后门攻击 cs.CL

16 pages, 6 figures

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13775v1) [paper-pdf](http://arxiv.org/pdf/2504.13775v1)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Ziwei Zhang, Yinghan Zhou, Yiming Xue

**Abstract**: Previous insertion-based and paraphrase-based backdoors have achieved great success in attack efficacy, but they ignore the text quality and semantic consistency between poisoned and clean texts. Although recent studies introduce LLMs to generate poisoned texts and improve the stealthiness, semantic consistency, and text quality, their hand-crafted prompts rely on expert experiences, facing significant challenges in prompt adaptability and attack performance after defenses. In this paper, we propose a novel backdoor attack based on adaptive optimization mechanism of black-box large language models (BadApex), which leverages a black-box LLM to generate poisoned text through a refined prompt. Specifically, an Adaptive Optimization Mechanism is designed to refine an initial prompt iteratively using the generation and modification agents. The generation agent generates the poisoned text based on the initial prompt. Then the modification agent evaluates the quality of the poisoned text and refines a new prompt. After several iterations of the above process, the refined prompt is used to generate poisoned texts through LLMs. We conduct extensive experiments on three dataset with six backdoor attacks and two defenses. Extensive experimental results demonstrate that BadApex significantly outperforms state-of-the-art attacks. It improves prompt adaptability, semantic consistency, and text quality. Furthermore, when two defense methods are applied, the average attack success rate (ASR) still up to 96.75%.

摘要: 之前的基于插入和基于转述的后门在攻击功效方面取得了巨大成功，但它们忽视了有毒文本和干净文本之间的文本质量和语义一致性。尽管最近的研究引入了LLM来生成有毒文本并提高隐蔽性、语义一致性和文本质量，但其手工制作的提示依赖于专家经验，在防御后的即时适应性和攻击性能方面面临着重大挑战。本文提出了一种基于黑匣子大型语言模型（BadApex）的自适应优化机制的新型后门攻击，该攻击利用黑匣子LLM通过细化的提示生成有毒文本。具体来说，自适应优化机制旨在使用生成和修改代理迭代地细化初始提示。生成代理根据初始提示生成中毒文本。然后修改代理评估中毒文本的质量并精炼新的提示。经过上述过程的多次迭代后，使用改进的提示通过LLM生成有毒文本。我们对三个数据集进行了广泛的实验，其中包含六种后门攻击和两种防御。大量实验结果表明，BadApex的性能明显优于最先进的攻击。它提高了即时适应性、语义一致性和文本质量。此外，当采用两种防御方法时，平均攻击成功率（ASB）仍高达96.75%。



## **2. Detecting Malicious Source Code in PyPI Packages with LLMs: Does RAG Come in Handy?**

使用LLM检测PyPI包中的恶意源代码：RAG方便吗？ cs.SE

The paper has been peer-reviewed and accepted for publication to the  29th International Conference on Evaluation and Assessment in Software  Engineering (EASE 2025)

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13769v1) [paper-pdf](http://arxiv.org/pdf/2504.13769v1)

**Authors**: Motunrayo Ibiyo, Thinakone Louangdy, Phuong T. Nguyen, Claudio Di Sipio, Davide Di Ruscio

**Abstract**: Malicious software packages in open-source ecosystems, such as PyPI, pose growing security risks. Unlike traditional vulnerabilities, these packages are intentionally designed to deceive users, making detection challenging due to evolving attack methods and the lack of structured datasets. In this work, we empirically evaluate the effectiveness of Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and few-shot learning for detecting malicious source code. We fine-tune LLMs on curated datasets and integrate YARA rules, GitHub Security Advisories, and malicious code snippets with the aim of enhancing classification accuracy. We came across a counterintuitive outcome: While RAG is expected to boost up the prediction performance, it fails in the performed evaluation, obtaining a mediocre accuracy. In contrast, few-shot learning is more effective as it significantly improves the detection of malicious code, achieving 97% accuracy and 95% balanced accuracy, outperforming traditional RAG approaches. Thus, future work should expand structured knowledge bases, refine retrieval models, and explore hybrid AI-driven cybersecurity solutions.

摘要: 开源生态系统中的恶意软件包（例如PyPI）带来了越来越大的安全风险。与传统漏洞不同，这些包是故意设计来欺骗用户的，由于攻击方法不断发展和缺乏结构化数据集，检测变得具有挑战性。在这项工作中，我们根据经验评估了大型语言模型（LLM）、检索增强生成（RAG）和少量学习检测恶意源代码的有效性。我们对精心策划的数据集进行微调，并集成YARA规则、GitHub安全建议和恶意代码片段，旨在提高分类准确性。我们遇到了一个违反直觉的结果：虽然RAG有望提高预测性能，但它在执行的评估中失败了，获得了平庸的准确性。相比之下，少量学习更有效，因为它显著提高了恶意代码的检测，达到了97%的准确率和95%的平衡准确率，优于传统的RAG方法。因此，未来的工作应该扩展结构化知识库，完善检索模型，并探索混合人工智能驱动的网络安全解决方案。



## **3. DETAM: Defending LLMs Against Jailbreak Attacks via Targeted Attention Modification**

SEARCH：通过定向注意力修改保护LLM免受越狱攻击 cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13562v1) [paper-pdf](http://arxiv.org/pdf/2504.13562v1)

**Authors**: Yu Li, Han Jiang, Zhihua Wei

**Abstract**: With the widespread adoption of Large Language Models (LLMs), jailbreak attacks have become an increasingly pressing safety concern. While safety-aligned LLMs can effectively defend against normal harmful queries, they remain vulnerable to such attacks. Existing defense methods primarily rely on fine-tuning or input modification, which often suffer from limited generalization and reduced utility. To address this, we introduce DETAM, a finetuning-free defense approach that improves the defensive capabilities against jailbreak attacks of LLMs via targeted attention modification. Specifically, we analyze the differences in attention scores between successful and unsuccessful defenses to identify the attention heads sensitive to jailbreak attacks. During inference, we reallocate attention to emphasize the user's core intention, minimizing interference from attack tokens. Our experimental results demonstrate that DETAM outperforms various baselines in jailbreak defense and exhibits robust generalization across different attacks and models, maintaining its effectiveness even on in-the-wild jailbreak data. Furthermore, in evaluating the model's utility, we incorporated over-defense datasets, which further validate the superior performance of our approach. The code will be released immediately upon acceptance.

摘要: 随着大型语言模型（LLM）的广泛采用，越狱攻击已成为一个日益紧迫的安全问题。虽然安全一致的LLM可以有效地防御正常的有害查询，但它们仍然容易受到此类攻击。现有的防御方法主要依赖于微调或输入修改，这通常会受到通用性有限和实用性降低的影响。为了解决这个问题，我们引入了SEARCH，这是一种无微调的防御方法，通过有针对性的注意力修改来提高针对LLM越狱攻击的防御能力。具体来说，我们分析成功和不成功防御之间注意力分数的差异，以识别对越狱攻击敏感的注意力头。在推理过程中，我们重新分配注意力以强调用户的核心意图，最大限度地减少攻击令牌的干扰。我们的实验结果表明，DeliverM在越狱防御方面的表现优于各种基线，并且在不同的攻击和模型之间表现出强大的概括性，即使在野外越狱数据上也保持其有效性。此外，在评估模型的效用时，我们纳入了过度防御数据集，这进一步验证了我们方法的卓越性能。该代码将在接受后立即发布。



## **4. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

通过大语言模型进行对抗风格增强以实现稳健的假新闻检测 cs.CL

WWW'25 research track accepted

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2406.11260v3) [paper-pdf](http://arxiv.org/pdf/2406.11260v3)

**Authors**: Sungwon Park, Sungwon Han, Xing Xie, Jae-Gil Lee, Meeyoung Cha

**Abstract**: The spread of fake news harms individuals and presents a critical social challenge that must be addressed. Although numerous algorithmic and insightful features have been developed to detect fake news, many of these features can be manipulated with style-conversion attacks, especially with the emergence of advanced language models, making it more difficult to differentiate from genuine news. This study proposes adversarial style augmentation, AdStyle, designed to train a fake news detector that remains robust against various style-conversion attacks. The primary mechanism involves the strategic use of LLMs to automatically generate a diverse and coherent array of style-conversion attack prompts, enhancing the generation of particularly challenging prompts for the detector. Experiments indicate that our augmentation strategy significantly improves robustness and detection performance when evaluated on fake news benchmark datasets.

摘要: 假新闻的传播伤害了个人，并提出了必须解决的严重社会挑战。尽管已经开发了许多算法和有洞察力的功能来检测假新闻，但其中许多功能都可以通过风格转换攻击来操纵，特别是随着高级语言模型的出现，使其更难与真实新闻区分开来。这项研究提出了对抗性风格增强AdStyle，旨在训练假新闻检测器，该检测器在对抗各种风格转换攻击时保持稳健。主要机制涉及战略性地使用LLM来自动生成多样化且连贯的风格转换攻击提示阵列，从而增强检测器特别具有挑战性的提示的生成。实验表明，当对假新闻基准数据集进行评估时，我们的增强策略显着提高了鲁棒性和检测性能。



## **5. Large Language Models for Validating Network Protocol Parsers**

用于验证网络协议解析器的大型语言模型 cs.SE

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13515v1) [paper-pdf](http://arxiv.org/pdf/2504.13515v1)

**Authors**: Mingwei Zheng, Danning Xie, Xiangyu Zhang

**Abstract**: Network protocol parsers are essential for enabling correct and secure communication between devices. Bugs in these parsers can introduce critical vulnerabilities, including memory corruption, information leakage, and denial-of-service attacks. An intuitive way to assess parser correctness is to compare the implementation with its official protocol standard. However, this comparison is challenging because protocol standards are typically written in natural language, whereas implementations are in source code. Existing methods like model checking, fuzzing, and differential testing have been used to find parsing bugs, but they either require significant manual effort or ignore the protocol standards, limiting their ability to detect semantic violations. To enable more automated validation of parser implementations against protocol standards, we propose PARVAL, a multi-agent framework built on large language models (LLMs). PARVAL leverages the capabilities of LLMs to understand both natural language and code. It transforms both protocol standards and their implementations into a unified intermediate representation, referred to as format specifications, and performs a differential comparison to uncover inconsistencies. We evaluate PARVAL on the Bidirectional Forwarding Detection (BFD) protocol. Our experiments demonstrate that PARVAL successfully identifies inconsistencies between the implementation and its RFC standard, achieving a low false positive rate of 5.6%. PARVAL uncovers seven unique bugs, including five previously unknown issues.

摘要: 网络协议解析器对于实现设备之间正确、安全的通信至关重要。这些解析器中的错误可能会引入关键漏洞，包括内存损坏、信息泄露和拒绝服务攻击。评估解析器正确性的直观方法是将实现与其官方协议标准进行比较。然而，这种比较具有挑战性，因为协议标准通常是用自然语言编写的，而实现是用源代码编写的。模型检查、模糊化和差异测试等现有方法已被用来发现解析错误，但它们要么需要大量的手动工作，要么忽略协议标准，从而限制了它们检测语义违规的能力。为了能够根据协议标准对解析器实现进行更自动化的验证，我们提出了PARVAR，这是一个基于大型语言模型（LLM）的多代理框架。PARVAR利用LLM的功能来理解自然语言和代码。它将协议标准及其实现转换为统一的中间表示（称为格式规范），并执行差异比较以发现不一致之处。我们在双向转发检测（Illustrator）协议上评估PARVAR。我们的实验表明，PARVAR成功地识别了实现与其RFC标准之间的不一致之处，实现了5.6%的低假阳性率。PARVAR发现了七个独特的错误，其中包括五个以前未知的问题。



## **6. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

揭示一致大型语言模型内在的道德脆弱性 cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.05050v2) [paper-pdf](http://arxiv.org/pdf/2504.05050v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.

摘要: 大型语言模型（LLM）是人工通用智能的基础探索，但它们通过指令调整和偏好学习与人类价值观的一致只能实现表面的合规性。在这里，我们证明，预训练期间嵌入的有害知识在LLM参数记忆中作为不可磨灭的“黑暗模式”持续存在，逃避对齐保障措施，并在分布变化时的对抗诱导下重新浮出水面。在这项研究中，我们首先通过证明当前的对齐方法只产生知识集合中的局部“安全区域”来从理论上分析对齐LLM的内在道德脆弱性。相比之下，预先训练的知识仍然通过高可能性的对抗轨迹与有害概念保持全球联系。基于这一理论见解，我们通过在分布转移下采用语义一致诱导来从经验上验证我们的发现--一种通过优化的对抗提示系统性地绕过对齐约束的方法。这种理论和经验相结合的方法在23个最先进的对齐LLM中的19个（包括DeepSeek-R1和LLaMA-3）上实现了100%的攻击成功率，揭示了它们的普遍漏洞。



## **7. GraphQLer: Enhancing GraphQL Security with Context-Aware API Testing**

GraphQLer：通过上下文感知API测试增强GraphQL安全性 cs.CR

Publicly available on: https://github.com/omar2535/GraphQLer

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13358v1) [paper-pdf](http://arxiv.org/pdf/2504.13358v1)

**Authors**: Omar Tsai, Jianing Li, Tsz Tung Cheung, Lejing Huang, Hao Zhu, Jianrui Xiao, Iman Sharafaldin, Mohammad A. Tayebi

**Abstract**: GraphQL is an open-source data query and manipulation language for web applications, offering a flexible alternative to RESTful APIs. However, its dynamic execution model and lack of built-in security mechanisms expose it to vulnerabilities such as unauthorized data access, denial-of-service (DoS) attacks, and injections. Existing testing tools focus on functional correctness, often overlooking security risks stemming from query interdependencies and execution context. This paper presents GraphQLer, the first context-aware security testing framework for GraphQL APIs. GraphQLer constructs a dependency graph to analyze relationships among mutations, queries, and objects, capturing critical interdependencies. It chains related queries and mutations to reveal authentication and authorization flaws, access control bypasses, and resource misuse. Additionally, GraphQLer tracks internal resource usage to uncover data leakage, privilege escalation, and replay attack vectors. We assess GraphQLer on various GraphQL APIs, demonstrating improved testing coverage - averaging a 35% increase, with up to 84% in some cases - compared to top-performing baselines. Remarkably, this is achieved in less time, making GraphQLer suitable for time-sensitive contexts. GraphQLer also successfully detects a known CVE and potential vulnerabilities in large-scale production APIs. These results underline GraphQLer's utility in proactively securing GraphQL APIs through automated, context-aware vulnerability detection.

摘要: GraphQL是一种用于Web应用程序的开源数据查询和操作语言，提供了RESTful API的灵活替代方案。然而，它的动态执行模型和缺乏内置安全机制使其面临未经授权的数据访问、拒绝服务（DPS）攻击和注入等漏洞。现有的测试工具专注于功能正确性，通常忽视了由查询相互依赖性和执行上下文产生的安全风险。本文介绍了GraphQLer，这是第一个针对GraphQL API的上下文感知安全测试框架。GraphQLer构建了一个依赖图来分析变化、查询和对象之间的关系，捕获关键的相互依赖关系。它链接相关的查询和变化，以揭示身份验证和授权缺陷，访问控制绕过和资源滥用。此外，GraphQLer还跟踪内部资源使用情况，以发现数据泄漏、权限提升和重放攻击向量。我们在各种GraphQL API上对GraphQLer进行了评估，与性能最好的基线相比，测试覆盖率平均提高了35%，在某些情况下高达84%。值得注意的是，这可以在更短的时间内实现，使GraphQLer适合时间敏感的上下文。GraphQLer还成功检测到大规模生产API中的已知UTE和潜在漏洞。这些结果强调了GraphQLer通过自动化、上下文感知漏洞检测主动保护GraphQL API的实用性。



## **8. GraphAttack: Exploiting Representational Blindspots in LLM Safety Mechanisms**

GraphAttack：利用LLM安全机制中的代表性盲点 cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13052v1) [paper-pdf](http://arxiv.org/pdf/2504.13052v1)

**Authors**: Sinan He, An Wang

**Abstract**: Large Language Models (LLMs) have been equipped with safety mechanisms to prevent harmful outputs, but these guardrails can often be bypassed through "jailbreak" prompts. This paper introduces a novel graph-based approach to systematically generate jailbreak prompts through semantic transformations. We represent malicious prompts as nodes in a graph structure with edges denoting different transformations, leveraging Abstract Meaning Representation (AMR) and Resource Description Framework (RDF) to parse user goals into semantic components that can be manipulated to evade safety filters. We demonstrate a particularly effective exploitation vector by instructing LLMs to generate code that realizes the intent described in these semantic graphs, achieving success rates of up to 87% against leading commercial LLMs. Our analysis reveals that contextual framing and abstraction are particularly effective at circumventing safety measures, highlighting critical gaps in current safety alignment techniques that focus primarily on surface-level patterns. These findings provide insights for developing more robust safeguards against structured semantic attacks. Our research contributes both a theoretical framework and practical methodology for systematically stress-testing LLM safety mechanisms.

摘要: 大型语言模型（LLM）配备了安全机制来防止有害输出，但这些护栏通常可以通过“越狱”提示绕过。本文介绍了一种新颖的基于图形的方法，通过语义转换系统地生成越狱提示。我们将恶意提示表示为图结构中的节点，边缘表示不同的转换，利用抽象意义表示（MRC）和资源描述框架（RDF）将用户目标解析为可以操纵以规避安全过滤器的语义组件。我们通过指示LLM生成实现这些语义图中描述的意图的代码来展示一种特别有效的利用载体，与领先的商业LLM相比，成功率高达87%。我们的分析表明，上下文框架和抽象在规避安全措施方面特别有效，凸显了当前主要关注表面模式的安全对齐技术中的关键差距。这些发现为开发针对结构化语义攻击的更强大的保护措施提供了见解。我们的研究为系统性压力测试LLM安全机制提供了理论框架和实践方法论。



## **9. From Sands to Mansions: Towards Automated Cyberattack Emulation with Classical Planning and Large Language Models**

从金沙到豪宅：利用经典规划和大型语言模型实现自动网络攻击模拟 cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2407.16928v3) [paper-pdf](http://arxiv.org/pdf/2407.16928v3)

**Authors**: Lingzhi Wang, Zhenyuan Li, Yi Jiang, Zhengkai Wang, Zonghan Guo, Jiahui Wang, Yangyang Wei, Xiangmin Shen, Wei Ruan, Yan Chen

**Abstract**: As attackers continually advance their tools, skills, and techniques during cyberattacks - particularly in modern Advanced Persistence Threats (APT) campaigns - there is a pressing need for a comprehensive and up-to-date cyberattack dataset to support threat-informed defense and enable benchmarking of defense systems in both academia and commercial solutions. However, there is a noticeable scarcity of cyberattack datasets: recent academic studies continue to rely on outdated benchmarks, while cyberattack emulation in industry remains limited due to the significant human effort and expertise required. Creating datasets by emulating advanced cyberattacks presents several challenges, such as limited coverage of attack techniques, the complexity of chaining multiple attack steps, and the difficulty of realistically mimicking actual threat groups. In this paper, we introduce modularized Attack Action and Attack Action Linking Model as a structured way to organizing and chaining individual attack steps into multi-step cyberattacks. Building on this, we propose Aurora, a system that autonomously emulates cyberattacks using third-party attack tools and threat intelligence reports with the help of classical planning and large language models. Aurora can automatically generate detailed attack plans, set up emulation environments, and semi-automatically execute the attacks. We utilize Aurora to create a dataset containing over 1,000 attack chains. To our best knowledge, Aurora is the only system capable of automatically constructing such a large-scale cyberattack dataset with corresponding attack execution scripts and environments. Our evaluation further demonstrates that Aurora outperforms the previous similar work and even the most advanced generative AI models in cyberattack emulation. To support further research, we published the cyberattack dataset and will publish the source code of Aurora.

摘要: 随着攻击者在网络攻击期间不断改进他们的工具、技能和技术--特别是在现代高级持久性威胁（APT）活动中--迫切需要一个全面且最新的网络攻击数据集来支持基于威胁的防御，并实现学术界和商业解决方案中的防御系统基准测试。然而，网络攻击数据集明显稀缺：最近的学术研究继续依赖过时的基准，而由于需要大量的人力和专业知识，行业中的网络攻击模拟仍然有限。通过模拟高级网络攻击创建数据集带来了几个挑战，例如攻击技术的覆盖范围有限、链接多个攻击步骤的复杂性以及真实模拟实际威胁组的困难。本文中，我们引入了模块化的攻击动作和攻击动作链接模型，作为一种将各个攻击步骤组织和链接为多步骤网络攻击的结构化方法。在此基础上，我们提出了Aurora，这是一个在经典规划和大型语言模型的帮助下使用第三方攻击工具和威胁情报报告自主模拟网络攻击的系统。Aurora可以自动生成详细的攻击计划、设置模拟环境并半自动执行攻击。我们利用Aurora创建一个包含1，000多个攻击链的数据集。据我们所知，Aurora是唯一能够自动构建如此大规模网络攻击数据集以及相应的攻击执行脚本和环境的系统。我们的评估进一步表明，Aurora在网络攻击模拟方面的表现优于之前的类似工作，甚至优于最先进的生成式人工智能模型。为了支持进一步的研究，我们发布了网络攻击数据集，并将发布Aurora的源代码。



## **10. ControlNET: A Firewall for RAG-based LLM System**

Control NET：基于RAG的LLM系统的防火墙 cs.CR

Project Page: https://ai.zjuicsr.cn/firewall

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.09593v2) [paper-pdf](http://arxiv.org/pdf/2504.09593v2)

**Authors**: Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang, Cong Wang, Zhan Qin

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.

摘要: 检索增强生成（RAG）显着增强了大型语言模型（LLM）的事实准确性和领域适应性。这一进步使它们能够在医疗保健、金融和企业应用程序等敏感领域广泛部署。RAG通过整合外部知识来缓解幻觉，但也会带来隐私风险和安全风险，尤其是数据泄露风险和数据中毒风险。虽然最近的研究探索了即时注射和中毒攻击，但在控制入站和出站查询流以减轻这些威胁的全面研究方面仍然存在显着差距。在本文中，我们提出了一种人工智能防火墙Controller NET，旨在保护基于RAG的LLM系统免受这些漏洞的影响。ControlNET通过利用激活转变现象来检测对抗性查询并通过语义分歧减轻其影响来控制查询流。我们使用最先进的开源LLM（Llama 3、Vicuna和Mistral）对四个不同的基准数据集（包括Mmarco、HotpotQA、FinQA和MedalSys）进行全面实验。我们的结果表明，ControlNET在检测和缓解安全威胁同时保持系统无害性方面达到了超过0.909 AUROC。总的来说，ControlNET提供了一种有效、健壮、无害的防御机制，标志着基于RAG的LLM系统安全部署的重大进步。



## **11. PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization**

PR攻击：通过二层优化对大型语言模型中的检索增强生成进行协调的预算-RAG攻击 cs.CR

Accepted at SIGIR 2025

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.07717v2) [paper-pdf](http://arxiv.org/pdf/2504.07717v2)

**Authors**: Yang Jiao, Xiaodong Wang, Kai Yang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of applications, e.g., medical question-answering, mathematical sciences, and code generation. However, they also exhibit inherent limitations, such as outdated knowledge and susceptibility to hallucinations. Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to address these issues, but it also introduces new vulnerabilities. Recent efforts have focused on the security of RAG-based LLMs, yet existing attack methods face three critical challenges: (1) their effectiveness declines sharply when only a limited number of poisoned texts can be injected into the knowledge database, (2) they lack sufficient stealth, as the attacks are often detectable by anomaly detection systems, which compromises their effectiveness, and (3) they rely on heuristic approaches to generate poisoned texts, lacking formal optimization frameworks and theoretic guarantees, which limits their effectiveness and applicability. To address these issues, we propose coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack that introduces a small number of poisoned texts into the knowledge database while embedding a backdoor trigger within the prompt. When activated, the trigger causes the LLM to generate pre-designed responses to targeted queries, while maintaining normal behavior in other contexts. This ensures both high effectiveness and stealth. We formulate the attack generation process as a bilevel optimization problem leveraging a principled optimization framework to develop optimal poisoned texts and triggers. Extensive experiments across diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving a high attack success rate even with a limited number of poisoned texts and significantly improved stealth compared to existing methods.

摘要: 大型语言模型（LLM）已在广泛的应用程序中表现出出色的性能，例如，医学问答、数学科学和代码生成。然而，它们也表现出固有的局限性，例如过时的知识和幻觉的易感性。检索增强一代（RAG）已成为解决这些问题的一个有希望的范式，但它也引入了新的漏洞。最近的工作重点是基于RAG的LLM的安全性，但现有的攻击方法面临三个关键挑战：（1）当只能将有限数量的有毒文本注入知识数据库时，它们的有效性急剧下降，（2）它们缺乏足够的隐蔽性，因为异常检测系统通常可以检测到攻击，这损害了它们的有效性，（3）它们依赖启发式方法来生成有毒文本，缺乏正式的优化框架和理论保证，这限制了它们的有效性和适用性。为了解决这些问题，我们提出了协调的预算-RAG攻击（PR-攻击），这是一种新型的优化驱动攻击，它将少量有毒文本引入知识数据库，同时在提示内嵌入后门触发器。激活时，触发器会使LLM生成对目标查询的预先设计的响应，同时在其他上下文中保持正常行为。这确保了高效率和隐形性。我们将攻击生成过程制定为一个双层优化问题，利用有原则的优化框架来开发最佳的有毒文本和触发器。跨不同LLM和数据集的广泛实验证明了PR-Attack的有效性，即使在数量有限的有毒文本的情况下也能实现很高的攻击成功率，并且与现有方法相比，隐身性显着提高。



## **12. Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails**

LLM护栏中的快速注射和越狱检测 cs.CR

12 pages, 5 figures, 6 tables

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11168v2) [paper-pdf](http://arxiv.org/pdf/2504.11168v2)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.

摘要: 大型语言模型（LLM）护栏系统旨在防止即时注入和越狱攻击。然而，他们仍然容易受到逃避技术的影响。我们演示了两种通过传统的字符注入方法和算法对抗机器学习（ML）规避技术绕过LLM提示注入和越狱检测系统的方法。通过对六种主要保护系统（包括微软的Azure Promise Shield和Meta的Promise Guard）进行测试，我们表明这两种方法都可以用来逃避检测，同时保持对抗性效用，在某些情况下实现高达100%的逃避成功。此外，我们还证明，对手可以通过利用离线白盒模型计算的单词重要性排名来提高针对黑盒目标的攻击成功率（ASB）。我们的研究结果揭示了当前LLM保护机制中的漏洞，并强调了对更坚固的护栏系统的需求。



## **13. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

软提示威胁：通过嵌入空间攻击开源LLM中的安全一致和取消学习 cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2402.09063v2) [paper-pdf](http://arxiv.org/pdf/2402.09063v2)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.

摘要: 当前对LLM对抗鲁棒性的研究重点是自然语言空间中的离散输入操纵，其可以直接转移到闭源模型。然而，这种方法忽视了开源模型的稳定发展。随着开源模型功能的进步，确保其安全性也变得越来越重要。然而，针对利用完整模型访问的开源LLM量身定制的攻击在很大程度上仍然未被探索。我们解决了这一研究空白并提出了嵌入空间攻击，该攻击直接攻击输入令牌的连续嵌入表示。我们发现，嵌入空间攻击比离散攻击或模型微调更有效地规避模型对齐并触发有害行为。此外，我们在取消学习的背景下提出了一种新颖的威胁模型，并表明嵌入空间攻击可以从多个数据集和模型中未学习的LLM中提取据称已删除的信息。我们的研究结果强调将空间攻击嵌入到开源LLM中作为重要威胁模型。触发警告：附录包含LLM生成的带有暴力和骚扰的文本。



## **14. LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in Current Benchmarks**

LLM取消学习揭示了当前基准中强于预期的核心集效应 cs.CL

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.10185v2) [paper-pdf](http://arxiv.org/pdf/2504.10185v2)

**Authors**: Soumyadeep Pal, Changsheng Wang, James Diffenderfer, Bhavya Kailkhura, Sijia Liu

**Abstract**: Large language model unlearning has become a critical challenge in ensuring safety and controlled model behavior by removing undesired data-model influences from the pretrained model while preserving general utility. Significant recent efforts have been dedicated to developing LLM unlearning benchmarks such as WMDP (Weapons of Mass Destruction Proxy) and MUSE (Machine Unlearning Six-way Evaluation), facilitating standardized unlearning performance assessment and method comparison. Despite their usefulness, we uncover for the first time a novel coreset effect within these benchmarks. Specifically, we find that LLM unlearning achieved with the original (full) forget set can be effectively maintained using a significantly smaller subset (functioning as a "coreset"), e.g., as little as 5% of the forget set, even when selected at random. This suggests that LLM unlearning in these benchmarks can be performed surprisingly easily, even in an extremely low-data regime. We demonstrate that this coreset effect remains strong, regardless of the LLM unlearning method used, such as NPO (Negative Preference Optimization) and RMU (Representation Misdirection Unlearning), the popular ones in these benchmarks. The surprisingly strong coreset effect is also robust across various data selection methods, ranging from random selection to more sophisticated heuristic approaches. We explain the coreset effect in LLM unlearning through a keyword-based perspective, showing that keywords extracted from the forget set alone contribute significantly to unlearning effectiveness and indicating that current unlearning is driven by a compact set of high-impact tokens rather than the entire dataset. We further justify the faithfulness of coreset-unlearned models along additional dimensions, such as mode connectivity and robustness to jailbreaking attacks. Codes are available at https://github.com/OPTML-Group/MU-Coreset.

摘要: 大型语言模型取消学习已成为通过从预训练模型中消除不希望的数据模型影响同时保持通用性来确保安全性和受控模型行为的一个关键挑战。最近做出了重大努力，致力于开发LLM忘记学习基准，例如WMDP（大规模杀伤性武器代理）和MUSE（机器忘记学习六路评估），促进标准化忘记学习性能评估和方法比较。尽管它们很有用，但我们首次在这些基准中发现了一种新颖的核心重置效应。具体来说，我们发现用原始（完整）忘记集实现的LLM取消学习可以使用明显较小的子集（充当“核心集”）有效地维护，例如，即使是随机选择，也只有忘记集的5%。这表明，即使在数据量极低的情况下，这些基准中的LLM取消学习也可以非常容易地执行。我们证明，无论使用何种LLM取消学习方法，例如NPO（负偏好优化）和RMU（代表误导取消学习），这种核心重置效应仍然很强，这些基准中流行的方法。令人惊讶的强烈核心集效应在各种数据选择方法中也很强大，从随机选择到更复杂的启发式方法。我们通过基于关键词的角度解释了LLM取消学习中的核心重置效应，表明仅从忘记集中提取的关键词对取消学习有效性做出了显着贡献，并表明当前的取消学习是由一组紧凑的高影响力令牌驱动的，而不是整个数据集。我们从其他维度（例如模式连接性和对越狱攻击的鲁棒性）进一步证明了未学习核心集的模型的忠实性。代码可访问https://github.com/OPTML-Group/MU-Coreset。



## **15. Entropy-Guided Watermarking for LLMs: A Test-Time Framework for Robust and Traceable Text Generation**

LLM的信息引导水印：用于稳健且可追溯的文本生成的测试时框架 cs.CL

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.12108v1) [paper-pdf](http://arxiv.org/pdf/2504.12108v1)

**Authors**: Shizhan Cai, Liang Ding, Dacheng Tao

**Abstract**: The rapid development of Large Language Models (LLMs) has intensified concerns about content traceability and potential misuse. Existing watermarking schemes for sampled text often face trade-offs between maintaining text quality and ensuring robust detection against various attacks. To address these issues, we propose a novel watermarking scheme that improves both detectability and text quality by introducing a cumulative watermark entropy threshold. Our approach is compatible with and generalizes existing sampling functions, enhancing adaptability. Experimental results across multiple LLMs show that our scheme significantly outperforms existing methods, achieving over 80\% improvements on widely-used datasets, e.g., MATH and GSM8K, while maintaining high detection accuracy.

摘要: 大型语言模型（LLM）的快速发展加剧了人们对内容可追溯性和潜在滥用的担忧。现有的采样文本水印方案经常面临保持文本质量和确保针对各种攻击的鲁棒检测之间的权衡。为了解决这些问题，我们提出了一种新颖的水印方案，通过引入累积水印信息阈值来提高可检测性和文本质量。我们的方法与现有的采样功能兼容并推广，增强了适应性。多个LLM的实验结果表明，我们的方案显着优于现有方法，在广泛使用的数据集上实现了超过80%的改进，例如MATH和GSM 8 K，同时保持高检测准确性。



## **16. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

代理安全工作台（ASB）：对基于LLM的代理中的攻击和防御进行形式化和基准化 cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2410.02644v3) [paper-pdf](http://arxiv.org/pdf/2410.02644v3)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.

摘要: 尽管基于LLM的代理在大型语言模型（LLM）的支持下可以使用外部工具和内存机制来解决复杂的现实世界任务，但它们也可能引入关键的安全漏洞。然而，现有文献并未全面评估针对基于LLM的代理的攻击和防御。为了解决这个问题，我们引入了代理安全工作台（ASB），这是一个全面的框架，旨在形式化、基准化和评估基于LLM的代理的攻击和防御，包括10种场景（例如，电子商务、自动驾驶、金融）、10个针对场景的代理、400多种工具、27种不同类型的攻击/防御方法和7个评估指标。基于ASB，我们对10种提示注入攻击、一种记忆中毒攻击、一种新颖的思想计划后门攻击、4种混合攻击以及13个LLM主干上的11种相应防御进行了基准测试。我们的基准测试结果揭示了代理操作不同阶段的关键漏洞，包括系统提示、用户提示处理、工具使用和内存检索，平均攻击成功率最高，为84.30%，但当前防御中表现出的有效性有限，揭示了社区在代理安全方面需要做的重要工作。我们还引入了一个新的指标来评估代理平衡实用性和安全性的能力。我们的代码可在https://github.com/agiresearch/ASB上找到。



## **17. On the Feasibility of Using MultiModal LLMs to Execute AR Social Engineering Attacks**

使用多模式LLM执行AR社会工程攻击的可行性 cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.13209v1) [paper-pdf](http://arxiv.org/pdf/2504.13209v1)

**Authors**: Ting Bi, Chenghang Ye, Zheyu Yang, Ziyi Zhou, Cui Tang, Jun Zhang, Zui Tao, Kailong Wang, Liting Zhou, Yang Yang, Tianlong Yu

**Abstract**: Augmented Reality (AR) and Multimodal Large Language Models (LLMs) are rapidly evolving, providing unprecedented capabilities for human-computer interaction. However, their integration introduces a new attack surface for social engineering. In this paper, we systematically investigate the feasibility of orchestrating AR-driven Social Engineering attacks using Multimodal LLM for the first time, via our proposed SEAR framework, which operates through three key phases: (1) AR-based social context synthesis, which fuses Multimodal inputs (visual, auditory and environmental cues); (2) role-based Multimodal RAG (Retrieval-Augmented Generation), which dynamically retrieves and integrates contextual data while preserving character differentiation; and (3) ReInteract social engineering agents, which execute adaptive multiphase attack strategies through inference interaction loops. To verify SEAR, we conducted an IRB-approved study with 60 participants in three experimental configurations (unassisted, AR+LLM, and full SEAR pipeline) compiling a new dataset of 180 annotated conversations in simulated social scenarios. Our results show that SEAR is highly effective at eliciting high-risk behaviors (e.g., 93.3% of participants susceptible to email phishing). The framework was particularly effective in building trust, with 85% of targets willing to accept an attacker's call after an interaction. Also, we identified notable limitations such as ``occasionally artificial'' due to perceived authenticity gaps. This work provides proof-of-concept for AR-LLM driven social engineering attacks and insights for developing defensive countermeasures against next-generation augmented reality threats.

摘要: 增强现实（AR）和多模式大型语言模型（LLM）正在迅速发展，为人机交互提供了前所未有的能力。然而，它们的集成为社会工程引入了新的攻击面。本文通过我们提出的SEAR框架，首次系统地研究了使用多模式LLM策划AR驱动的社会工程攻击的可行性，该框架通过三个关键阶段运行：（1）基于AR的社会上下文合成，融合了多模式输入（视觉、听觉和环境线索）;（2）基于角色的多模式RAG（检索-增强代），动态检索和集成上下文数据，同时保留字符差异;和（3）ReInteract社会工程代理，通过推理交互循环执行自适应多阶段攻击策略。为了验证SEAR，我们对60名参与者进行了一项获得IRC批准的研究，参与者分为三种实验配置（无辅助、AR+LLM和完整SEAR管道），在模拟社交场景中编制了一个包含180个注释对话的新数据集。我们的结果表明，SEAR在引发高风险行为（例如，93.3%的参与者容易受到电子邮件网络钓鱼）。该框架在建立信任方面特别有效，85%的目标愿意在互动后接受攻击者的电话。此外，我们还发现了由于感知到的真实性差距而存在的显着局限性，例如“偶尔是人为的”。这项工作为AR-LLM驱动的社会工程攻击提供了概念验证，并为开发针对下一代增强现实威胁的防御对策提供了见解。



## **18. Progent: Programmable Privilege Control for LLM Agents**

Progent：LLM代理的可编程特权控制 cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11703v1) [paper-pdf](http://arxiv.org/pdf/2504.11703v1)

**Authors**: Tianneng Shi, Jingxuan He, Zhun Wang, Linyu Wu, Hongwei Li, Wenbo Guo, Dawn Song

**Abstract**: LLM agents are an emerging form of AI systems where large language models (LLMs) serve as the central component, utilizing a diverse set of tools to complete user-assigned tasks. Despite their great potential, LLM agents pose significant security risks. When interacting with the external world, they may encounter malicious commands from attackers, leading to the execution of dangerous actions. A promising way to address this is by enforcing the principle of least privilege: allowing only essential actions for task completion while blocking unnecessary ones. However, achieving this is challenging, as it requires covering diverse agent scenarios while preserving both security and utility.   We introduce Progent, the first privilege control mechanism for LLM agents. At its core is a domain-specific language for flexibly expressing privilege control policies applied during agent execution. These policies provide fine-grained constraints over tool calls, deciding when tool calls are permissible and specifying fallbacks if they are not. This enables agent developers and users to craft suitable policies for their specific use cases and enforce them deterministically to guarantee security. Thanks to its modular design, integrating Progent does not alter agent internals and requires only minimal changes to agent implementation, enhancing its practicality and potential for widespread adoption. To automate policy writing, we leverage LLMs to generate policies based on user queries, which are then updated dynamically for improved security and utility. Our extensive evaluation shows that it enables strong security while preserving high utility across three distinct scenarios or benchmarks: AgentDojo, ASB, and AgentPoison. Furthermore, we perform an in-depth analysis, showcasing the effectiveness of its core components and the resilience of its automated policy generation against adaptive attacks.

摘要: LLM代理是人工智能系统的一种新兴形式，其中大型语言模型（LLM）作为中心组件，利用一组不同的工具来完成用户分配的任务。尽管LLM代理潜力巨大，但仍构成重大安全风险。在与外部世界互动时，他们可能会遇到攻击者的恶意命令，导致执行危险动作。解决这个问题的一个有希望的方法是执行最小特权原则：仅允许执行完成任务的必要动作，同时阻止不必要的动作。然而，实现这一点具有挑战性，因为它需要覆盖不同的代理场景，同时保持安全性和实用性。   我们引入Progent，这是LLM代理的第一个特权控制机制。其核心是一种特定于领域的语言，用于灵活表达代理执行期间应用的特权控制策略。这些策略为工具调用提供了细粒度的约束，决定何时允许工具调用，并在不允许时指定后备。这使代理开发人员和用户能够为他们的特定用例制定合适的策略，并确定性地实施这些策略以保证安全性。由于其模块化设计，集成Progent不会改变代理的内部结构，只需要对代理的实现进行最小的更改，从而增强了其实用性和广泛采用的潜力。为了自动化策略编写，我们利用LLM根据用户查询生成策略，然后动态更新以提高安全性和实用性。我们的广泛评估表明，它可以实现强大的安全性，同时在三个不同的场景或基准测试中保持高实用性：AgentDojo，ASB和AgentPoison。此外，我们还进行了深入的分析，展示了其核心组件的有效性以及自动化策略生成针对自适应攻击的弹性。



## **19. Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' "Typo" Correction**

通过LLM辅助频谱图的“错别字”纠正，使对噪音键盘的声学侧通道攻击变得可行 cs.CR

Length: 13 pages Figures: 5 figures Tables: 7 tables Keywords:  Acoustic side-channel attacks, machine learning, Visual Transformers, Large  Language Models (LLMs), security Conference: Accepted at the 19th USENIX WOOT  Conference on Offensive Technologies (WOOT '25). Licensing: This paper is  submitted under the CC BY Creative Commons Attribution license. arXiv admin  note: text overlap with arXiv:2502.09782

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11622v1) [paper-pdf](http://arxiv.org/pdf/2504.11622v1)

**Authors**: Seyyed Ali Ayati, Jin Hyun Park, Yichen Cai, Marcus Botacin

**Abstract**: The large integration of microphones into devices increases the opportunities for Acoustic Side-Channel Attacks (ASCAs), as these can be used to capture keystrokes' audio signals that might reveal sensitive information. However, the current State-Of-The-Art (SOTA) models for ASCAs, including Convolutional Neural Networks (CNNs) and hybrid models, such as CoAtNet, still exhibit limited robustness under realistic noisy conditions. Solving this problem requires either: (i) an increased model's capacity to infer contextual information from longer sequences, allowing the model to learn that an initially noisily typed word is the same as a futurely collected non-noisy word, or (ii) an approach to fix misidentified information from the contexts, as one does not type random words, but the ones that best fit the conversation context. In this paper, we demonstrate that both strategies are viable and complementary solutions for making ASCAs practical. We observed that no existing solution leverages advanced transformer architectures' power for these tasks and propose that: (i) Visual Transformers (VTs) are the candidate solutions for capturing long-term contextual information and (ii) transformer-powered Large Language Models (LLMs) are the candidate solutions to fix the ``typos'' (mispredictions) the model might make. Thus, we here present the first-of-its-kind approach that integrates VTs and LLMs for ASCAs.   We first show that VTs achieve SOTA performance in classifying keystrokes when compared to the previous CNN benchmark. Second, we demonstrate that LLMs can mitigate the impact of real-world noise. Evaluations on the natural sentences revealed that: (i) incorporating LLMs (e.g., GPT-4o) in our ASCA pipeline boosts the performance of error-correction tasks; and (ii) the comparable performance can be attained by a lightweight, fine-tuned smaller LLM (67 times smaller than GPT-4o), using...

摘要: 麦克风大量集成到设备中增加了声学侧道攻击（ASCA）的机会，因为这些攻击可用于捕获可能泄露敏感信息的麦克风音频信号。然而，当前ASCA的最新技术水平（SOTA）模型（包括卷积神经网络（CNN）和混合模型（例如CoAtNet））在现实噪音条件下仍然表现出有限的鲁棒性。解决这个问题需要：（i）提高模型从更长的序列中推断上下文信息的能力，允许模型学习最初输入的有噪音的单词与未来收集的无噪音单词相同，或者（ii）修复来自上下文的错误识别信息的方法，因为输入的不是随机单词，而是最适合对话上下文的单词。在本文中，我们证明了这两种策略都是使ASCA实用的可行且相辅相成的解决方案。我们观察到，没有现有的解决方案利用高级Transformer架构的能力来完成这些任务，并建议：（i）视觉转换器（VT）是捕获长期上下文信息的候选解决方案，（ii）转换器驱动的大型语言模型（LLM）是修复模型可能造成的“拼写错误”（预测错误）的候选解决方案。因此，我们在这里介绍了一种首创的方法，该方法将VT和LLM集成到ASCA中。   我们首先表明，与之前的CNN基准相比，VT在分类击键方面实现了SOTA性能。其次，我们证明LLM可以减轻现实世界噪音的影响。对自然句子的评估显示：（i）纳入LLM（例如，我们的ASCA管道中的GPT-4 o）提高了错误纠正任务的性能;并且（ii）通过轻量级、微调的较小LLM（比GPT-4 o小67倍）可以获得相当的性能，使用.



## **20. Propaganda via AI? A Study on Semantic Backdoors in Large Language Models**

通过人工智能进行宣传？大型语言模型中的语义后门研究 cs.CL

18 pages, 1 figure

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.12344v1) [paper-pdf](http://arxiv.org/pdf/2504.12344v1)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Large language models (LLMs) demonstrate remarkable performance across myriad language tasks, yet they remain vulnerable to backdoor attacks, where adversaries implant hidden triggers that systematically manipulate model outputs. Traditional defenses focus on explicit token-level anomalies and therefore overlook semantic backdoors-covert triggers embedded at the conceptual level (e.g., ideological stances or cultural references) that rely on meaning-based cues rather than lexical oddities. We first show, in a controlled finetuning setting, that such semantic backdoors can be implanted with only a small poisoned corpus, establishing their practical feasibility. We then formalize the notion of semantic backdoors in LLMs and introduce a black-box detection framework, RAVEN (short for "Response Anomaly Vigilance for uncovering semantic backdoors"), which combines semantic entropy with cross-model consistency analysis. The framework probes multiple models with structured topic-perspective prompts, clusters the sampled responses via bidirectional entailment, and flags anomalously uniform outputs; cross-model comparison isolates model-specific anomalies from corpus-wide biases. Empirical evaluations across diverse LLM families (GPT-4o, Llama, DeepSeek, Mistral) uncover previously undetected semantic backdoors, providing the first proof-of-concept evidence of these hidden vulnerabilities and underscoring the urgent need for concept-level auditing of deployed language models. We open-source our code and data at https://github.com/NayMyatMin/RAVEN.

摘要: 大型语言模型（LLM）在无数语言任务中表现出出色的性能，但它们仍然容易受到后门攻击，即对手植入隐藏触发器来系统性地操纵模型输出。传统防御专注于显式标记级异常，因此忽视了嵌入在概念级的语义后门隐蔽触发器（例如，意识形态立场或文化参考）依赖于基于意义的线索，而不是词汇上的怪异。我们首先表明，在受控微调环境中，这种语义后门只能植入一个小的有毒主体，从而建立了它们的实际可行性。然后，我们在LLM中形式化了语义后门的概念，并引入了黑匣子检测框架RAVEN（“揭露语义后门的响应异常警戒”的缩写），该框架将语义熵与跨模型一致性分析相结合。该框架通过结构化的主题视角提示来探索多个模型，通过双向蕴含对采样的响应进行聚集，并标记出极其均匀的输出;跨模型比较将模型特定的异常与整个群体的偏差隔离开来。对不同LLM家族（GPT-4 o、Llama、DeepSeek、Mistral）的经验评估揭示了之前未检测到的语义后门，提供了这些隐藏漏洞的第一个概念验证证据，并强调了对已部署语言模型进行概念级审计的迫切需要。我们在https://github.com/NayMyatMin/RAVEN上开源我们的代码和数据。



## **21. Lateral Phishing With Large Language Models: A Large Organization Comparative Study**

大型语言模型的横向网络钓鱼：大型组织比较研究 cs.CR

Accepted for publication in IEEE Access. This version includes  revisions following peer review

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2401.09727v2) [paper-pdf](http://arxiv.org/pdf/2401.09727v2)

**Authors**: Mazal Bethany, Athanasios Galiopoulos, Emet Bethany, Mohammad Bahrami Karkevandi, Nicole Beebe, Nishant Vishwamitra, Peyman Najafirad

**Abstract**: The emergence of Large Language Models (LLMs) has heightened the threat of phishing emails by enabling the generation of highly targeted, personalized, and automated attacks. Traditionally, many phishing emails have been characterized by typos, errors, and poor language. These errors can be mitigated by LLMs, potentially lowering the barrier for attackers. Despite this, there is a lack of large-scale studies comparing the effectiveness of LLM-generated lateral phishing emails to those crafted by humans. Current literature does not adequately address the comparative effectiveness of LLM and human-generated lateral phishing emails in a real-world, large-scale organizational setting, especially considering the potential for LLMs to generate more convincing and error-free phishing content. To address this gap, we conducted a pioneering study within a large university, targeting its workforce of approximately 9,000 individuals including faculty, staff, administrators, and student workers. Our results indicate that LLM-generated lateral phishing emails are as effective as those written by communications professionals, emphasizing the critical threat posed by LLMs in leading phishing campaigns. We break down the results of the overall phishing experiment, comparing vulnerability between departments and job roles. Furthermore, to gather qualitative data, we administered a detailed questionnaire, revealing insights into the reasons and motivations behind vulnerable employee's actions. This study contributes to the understanding of cyber security threats in educational institutions and provides a comprehensive comparison of LLM and human-generated phishing emails' effectiveness, considering the potential for LLMs to generate more convincing content. The findings highlight the need for enhanced user education and system defenses to mitigate the growing threat of AI-powered phishing attacks.

摘要: 大型语言模型（LLM）的出现通过生成高度针对性、个性化和自动化的攻击，加剧了网络钓鱼电子邮件的威胁。传统上，许多网络钓鱼电子邮件的特点是拼写错误、错误和语言拙劣。这些错误可以通过LLM来缓解，从而可能降低攻击者的障碍。尽管如此，缺乏大规模研究将LLM生成的横向网络钓鱼电子邮件与人类制作的横向网络钓鱼电子邮件的有效性进行比较。当前的文献没有充分解决LLM和人类生成的横向网络钓鱼电子邮件在现实世界、大规模组织环境中的比较有效性，特别是考虑到LLM生成更令人信服且无错误的网络钓鱼内容的潜力。为了解决这一差距，我们在一所大型大学内进行了一项开创性研究，目标是其约9，000名员工，包括教职员工、管理人员和学生工作者。我们的结果表明，LLM生成的横向网络钓鱼电子邮件与通信专业人士撰写的电子邮件一样有效，强调了LLM在领先的网络钓鱼活动中构成的严重威胁。我们分解了整个网络钓鱼实验的结果，比较了部门和工作角色之间的脆弱性。此外，为了收集定性数据，我们进行了一份详细的调查问卷，揭示了对弱势员工行为背后的原因和动机的见解。这项研究有助于了解教育机构的网络安全威胁，并对LLM和人类生成的网络钓鱼电子邮件的有效性进行了全面比较，同时考虑到LLM生成更令人信服的内容的潜力。研究结果凸显了加强用户教育和系统防御的必要性，以减轻人工智能驱动的网络钓鱼攻击日益严重的威胁。



## **22. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

SCA：高效语义一致的无限制对抗攻击 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2410.02240v5) [paper-pdf](http://arxiv.org/pdf/2410.02240v5)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.

摘要: 部署在敏感环境中的基于深度神经网络的系统很容易受到对抗攻击。不受限制的对抗攻击通常操纵图像的语义内容（例如，颜色或纹理）来创建既有效又逼真的对抗示例。最近的作品利用扩散倒置过程将图像映射到潜在空间，其中通过引入扰动来操纵高级语义。然而，它们通常会导致去噪输出中出现严重的语义扭曲，并且效率低下。在这项研究中，我们提出了一种名为语义一致的无限制对抗攻击（SCA）的新型框架，该框架采用倒置方法来提取编辑友好的噪音图，并利用多模式大型语言模型（MLLM）在整个过程中提供语义指导。在MLLM提供丰富的语义信息的情况下，我们使用一系列编辑友好的噪音图来执行每一步的DDPM去噪过程，并利用DeliverSolver ++加速这一过程，实现具有语义一致性的高效采样。与现有方法相比，我们的框架能够高效生成表现出最小可辨别的语义变化的对抗性示例。因此，我们首次引入语义一致的对抗示例（SCAE）。大量的实验和可视化已经证明了SCA的高效率，特别是平均比最先进的攻击快12倍。本文的研究可以进一步引起人们对多媒体信息安全的关注。



## **23. The Obvious Invisible Threat: LLM-Powered GUI Agents' Vulnerability to Fine-Print Injections**

显而易见的不可见威胁：LLM-Powered GUI代理对Fine-Print注入的脆弱性 cs.HC

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11281v1) [paper-pdf](http://arxiv.org/pdf/2504.11281v1)

**Authors**: Chaoran Chen, Zhiping Zhang, Bingcan Guo, Shang Ma, Ibrahim Khalilov, Simret A Gebreegziabher, Yanfang Ye, Ziang Xiao, Yaxing Yao, Tianshi Li, Toby Jia-Jun Li

**Abstract**: A Large Language Model (LLM) powered GUI agent is a specialized autonomous system that performs tasks on the user's behalf according to high-level instructions. It does so by perceiving and interpreting the graphical user interfaces (GUIs) of relevant apps, often visually, inferring necessary sequences of actions, and then interacting with GUIs by executing the actions such as clicking, typing, and tapping. To complete real-world tasks, such as filling forms or booking services, GUI agents often need to process and act on sensitive user data. However, this autonomy introduces new privacy and security risks. Adversaries can inject malicious content into the GUIs that alters agent behaviors or induces unintended disclosures of private information. These attacks often exploit the discrepancy between visual saliency for agents and human users, or the agent's limited ability to detect violations of contextual integrity in task automation. In this paper, we characterized six types of such attacks, and conducted an experimental study to test these attacks with six state-of-the-art GUI agents, 234 adversarial webpages, and 39 human participants. Our findings suggest that GUI agents are highly vulnerable, particularly to contextually embedded threats. Moreover, human users are also susceptible to many of these attacks, indicating that simple human oversight may not reliably prevent failures. This misalignment highlights the need for privacy-aware agent design. We propose practical defense strategies to inform the development of safer and more reliable GUI agents.

摘要: 由大型语言模型（LLM）驱动的图形用户界面代理是一个专门的自治系统，根据高级指令代表用户执行任务。它通过感知和解释相关应用程序的图形用户界面（GUIs）（通常是视觉上的），推断必要的操作序列，然后通过执行单击、打字和点击等操作与GUIs交互来实现这一目标。为了完成现实世界的任务，例如填写表格或预订服务，图形用户界面代理通常需要处理和处理敏感用户数据。然而，这种自主性带来了新的隐私和安全风险。对手可以将恶意内容注入图形用户界面，从而改变代理行为或导致私人信息的意外泄露。这些攻击通常利用代理和人类用户的视觉显著性之间的差异，或者代理检测任务自动化中上下文完整性违规的能力有限。在本文中，我们描述了六种类型的此类攻击，并进行了一项实验研究，使用六个最先进的图形用户界面代理、234个对抗性网页和39名人类参与者来测试这些攻击。我们的研究结果表明，图形用户界面代理非常容易受到攻击，特别是对于上下文嵌入式威胁。此外，人类用户也容易受到许多此类攻击，这表明简单的人类监督可能无法可靠地防止故障。这种错位凸显了隐私感知代理设计的必要性。我们提出了实用的防御策略，为开发更安全、更可靠的图形用户界面代理提供信息。



## **24. Exploring Backdoor Attack and Defense for LLM-empowered Recommendations**

探索LLM授权建议的后门攻击和防御 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11182v1) [paper-pdf](http://arxiv.org/pdf/2504.11182v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner.

摘要: 大型语言模型（LLM）与推荐系统（RecSys）的融合极大地提高了个性化推荐并引起了广泛关注。尽管取得了令人印象深刻的进展，但基于LLM的RecSys抵御后门攻击的安全性在很大程度上仍然没有得到充分的探索。在本文中，我们提出了一个新问题：具有特定触发器的后门是否会被注入到基于LLM的Recsys中，从而导致当后门触发器附加到项目标题时推荐响应的操纵？为了调查基于LLM的RecSys在后门攻击下的漏洞，我们提出了一种新的攻击框架，称为RecSys后门注入中毒（BadRec）。BadRec通过触发器扰乱这些物品的标题，并雇用几名虚假用户与这些物品互动，有效地毒害了训练集，并为基于LLM的RecSys注入后门。全面的实验表明，仅用对抗性示例毒害1%的训练数据就足以成功植入后门，从而能够操纵推荐。为了进一步减轻此类安全威胁，我们提出了一种名为毒药扫描仪（P-Scanner）的通用防御策略。具体来说，我们引入了基于LLM的毒物扫描仪，通过利用LLM强大的语言理解能力和丰富的知识来检测有毒物品。触发增强代理被用来生成不同的合成触发器，以引导中毒扫描器学习中毒物品检测任务的特定于领域的知识。在三个真实数据集上的大量实验验证了所提出的P-Scanner的有效性。



## **25. QAVA: Query-Agnostic Visual Attack to Large Vision-Language Models**

QAVA：对大型视觉语言模型的查询不可知视觉攻击 cs.CV

Accepted by NAACL 2025 main

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11038v1) [paper-pdf](http://arxiv.org/pdf/2504.11038v1)

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Yu Wang

**Abstract**: In typical multimodal tasks, such as Visual Question Answering (VQA), adversarial attacks targeting a specific image and question can lead large vision-language models (LVLMs) to provide incorrect answers. However, it is common for a single image to be associated with multiple questions, and LVLMs may still answer other questions correctly even for an adversarial image attacked by a specific question. To address this, we introduce the query-agnostic visual attack (QAVA), which aims to create robust adversarial examples that generate incorrect responses to unspecified and unknown questions. Compared to traditional adversarial attacks focused on specific images and questions, QAVA significantly enhances the effectiveness and efficiency of attacks on images when the question is unknown, achieving performance comparable to attacks on known target questions. Our research broadens the scope of visual adversarial attacks on LVLMs in practical settings, uncovering previously overlooked vulnerabilities, particularly in the context of visual adversarial threats. The code is available at https://github.com/btzyd/qava.

摘要: 在典型的多模式任务中，例如视觉问题解答（VQA），针对特定图像和问题的对抗攻击可能会导致大型视觉语言模型（LVLM）提供错误的答案。然而，单个图像与多个问题关联是常见的，即使对于受到特定问题攻击的对抗图像，LVLM仍然可以正确回答其他问题。为了解决这个问题，我们引入了查询不可知视觉攻击（QAVA），其目的是创建强大的对抗性示例，这些示例会对未指定和未知的问题生成错误的响应。与针对特定图像和问题的传统对抗攻击相比，QAVA显着增强了问题未知时图像攻击的有效性和效率，实现了与针对已知目标问题的攻击相当的性能。我们的研究扩大了实际环境中对LVLM的视觉对抗攻击的范围，揭示了以前被忽视的漏洞，特别是在视觉对抗威胁的背景下。该代码可在https://github.com/btzyd/qava上获取。



## **26. Concept Enhancement Engineering: A Lightweight and Efficient Robust Defense Against Jailbreak Attacks in Embodied AI**

概念增强工程：针对智能人工智能越狱攻击的轻量级、高效的稳健防御 cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.13201v1) [paper-pdf](http://arxiv.org/pdf/2504.13201v1)

**Authors**: Jirui Yang, Zheyu Lin, Shuhan Yang, Zhihui Lu, Xin Du

**Abstract**: Embodied Intelligence (EI) systems integrated with large language models (LLMs) face significant security risks, particularly from jailbreak attacks that manipulate models into generating harmful outputs or executing unsafe physical actions. Traditional defense strategies, such as input filtering and output monitoring, often introduce high computational overhead or interfere with task performance in real-time embodied scenarios. To address these challenges, we propose Concept Enhancement Engineering (CEE), a novel defense framework that leverages representation engineering to enhance the safety of embodied LLMs by dynamically steering their internal activations. CEE operates by (1) extracting multilingual safety patterns from model activations, (2) constructing control directions based on safety-aligned concept subspaces, and (3) applying subspace concept rotation to reinforce safe behavior during inference. Our experiments demonstrate that CEE effectively mitigates jailbreak attacks while maintaining task performance, outperforming existing defense methods in both robustness and efficiency. This work contributes a scalable and interpretable safety mechanism for embodied AI, bridging the gap between theoretical representation engineering and practical security applications. Our findings highlight the potential of latent-space interventions as a viable defense paradigm against emerging adversarial threats in physically grounded AI systems.

摘要: 与大型语言模型（LLM）集成的分布式智能（EI）系统面临着重大的安全风险，特别是来自操纵模型生成有害输出或执行不安全物理动作的越狱攻击。传统的防御策略，例如输入过滤和输出监控，通常会引入高计算负担或干扰实时具体场景中的任务性能。为了应对这些挑战，我们提出了概念增强工程（CEE），这是一种新型防御框架，它利用表示工程通过动态引导其内部激活来增强具体LLM的安全性。CEE的运作方式是：（1）从模型激活中提取多语言安全模式，（2）基于安全对齐的概念子空间构建控制方向，以及（3）应用子空间概念旋转来加强推理期间的安全行为。我们的实验表明，CEE有效地缓解了越狱攻击，同时保持了任务性能，在鲁棒性和效率方面都优于现有的防御方法。这项工作为嵌入式人工智能提供了可扩展和可解释的安全机制，弥合了理论表示工程和实际安全应用之间的差距。我们的研究结果强调了潜伏空间干预作为一种可行的防御范式的潜力，以应对物理基础人工智能系统中出现的对抗威胁。



## **27. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

迈向智能和安全的云：大型语言模型增强主动防御 cs.CR

7 pages; In submission

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2412.21051v2) [paper-pdf](http://arxiv.org/pdf/2412.21051v2)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Yuyu Zhao

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided a large number of benefits in daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks. Recent advancements in generative foundation models (GFMs), particularly in the large language models (LLMs), offer promising solutions for security intelligence. By exploiting the powerful abilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel proactive defense architecture that defeats various threats in a proactive manner. LLM-PD can efficiently make a decision through comprehensive data analysis and sequential reasoning, as well as dynamically creating and deploying actionable defense mechanisms on the target cloud. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. The experimental results demonstrate its remarkable ability in terms of defense effectiveness and efficiency, particularly highlighting an outstanding success rate when compared with other existing methods.

摘要: 云计算技术的快速发展和云应用程序数量的不断增加为日常生活带来了大量好处。然而，不同组件的多样性和复杂性对云安全构成了重大挑战，特别是在处理复杂和高级的网络攻击时。生成式基础模型（GFM）的最新进展，特别是大型语言模型（LLM），为安全智能提供了有前途的解决方案。通过利用语言理解、数据分析、任务推理、行动规划和代码生成方面的强大能力，我们提出了LLM-PD，这是一种新型的主动防御架构，可以以主动的方式击败各种威胁。LLM-PD可以通过全面的数据分析和顺序推理，以及在目标云上动态创建和部署可操作的防御机制来有效地做出决策。此外，它可以根据从之前的交互中学到的经验灵活地自我进化，并在无需额外训练的情况下适应新的攻击场景。实验结果证明了其在防御有效性和效率方面的出色能力，特别是与其他现有方法相比具有出色的成功率。



## **28. Adversarial Prompt Distillation for Vision-Language Models**

视觉语言模型的对抗性即时蒸馏 cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2411.15244v2) [paper-pdf](http://arxiv.org/pdf/2411.15244v2)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.

摘要: 对比图像预训练（CLIP）等大型预训练视觉语言模型（VLM）已被证明容易受到对抗攻击，这引发了人们对其在自动驾驶和医疗诊断等安全关键应用中部署的担忧。对抗性提示调整（APT）是对预训练的VLM进行鲁棒化的一种有希望的方法，它在提示调整的过程中应用对抗性训练。然而，现有的APT方法大多是单模式方法，仅为视觉或文本模式设计提示，从而限制了其稳健性或清晰准确性的有效性。在这项工作中，我们提出了对抗性提示蒸馏（APT），这是一个双峰知识蒸馏框架，通过将APT与多模式知识转移集成来增强APT。APT优化视觉和文本模式的提示，同时从干净的预培训教师CLIP模型中提取知识。对多个基准数据集的广泛实验证明了我们的APT方法在对抗稳健性和精确性方面优于当前最先进的APT方法。APT的有效性也验证了使用非稳健教师来提高微调后的VLM的通用性和稳健性的可能性。



## **29. Can LLMs handle WebShell detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework**

LLM可以处理WebShell检测吗？使用行为功能感知框架克服检测挑战 cs.CR

Under Review

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.13811v1) [paper-pdf](http://arxiv.org/pdf/2504.13811v1)

**Authors**: Feijiang Han, Jiaming Zhang, Chuyi Deng, Jianheng Tang, Yunhuai Liu

**Abstract**: WebShell attacks, in which malicious scripts are injected into web servers, are a major cybersecurity threat. Traditional machine learning and deep learning methods are hampered by issues such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models (LLMs) have gained attention for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two major contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling (WBFP) that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all models lag behind previous State-Of-The-Art (SOTA) methods. With BFAD, the performance of all LLMs improved, with an average F1 score increase of 13.82%. Larger models such as GPT-4, LLaMA 3.1 70B, and Qwen 2.5 14B outperform SOTA methods, while smaller models such as Qwen 2.5 3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection, and provides solutions to address the challenges in this task.

摘要: WebShell攻击（将恶意脚本注入网络服务器）是一种主要的网络安全威胁。传统的机器学习和深度学习方法受到需要大量训练数据、灾难性遗忘和概括性较差等问题的阻碍。最近，大型语言模型（LLM）在代码相关任务方面受到了关注，但它们在WebShell检测中的潜力仍然没有得到充分的探索。在本文中，我们做出了两个主要贡献：（1）对七种LLM进行全面评估，包括GPT-4、LLaMA 3.1 70 B和Qwen 2.5变体，使用26.59 K个PHP脚本的数据集以传统的基于序列和图形的方法为基准，以及（2）行为功能感知检测（BFAD）框架，旨在解决将LLM应用于该领域的具体挑战。我们的框架集成了三个组件：隔离恶意PHP函数调用的关键函数过滤器、捕获最具行为指示性的代码段的上下文感知代码提取策略，以及通过根据区分性功能级别配置文件优先考虑最相关的演示来增强上下文学习的加权行为函数剖析（WBFP）。我们的结果表明，较大的LLM可以实现近乎完美的精确度，但召回率较低，而较小的模型则表现出相反的权衡。然而，所有模型都落后于之前的最新技术水平（SOTA）方法。通过BFAD，所有LLM的表现都有所提高，F1平均得分提高了13.82%。GPT-4、LLaMA 3.1 70 B和Qwen 2.5 14 B等较大型号的性能优于SOTA方法，而Qwen 2.5 3B等较小型号的性能与传统方法具有竞争力。这项工作是首次探索LLM用于WebShell检测的可行性和局限性的工作，并提供了解决方案来应对这项任务中的挑战。



## **30. The Jailbreak Tax: How Useful are Your Jailbreak Outputs?**

越狱税：你的越狱输出有多有用？ cs.LG

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10694v1) [paper-pdf](http://arxiv.org/pdf/2504.10694v1)

**Authors**: Kristina Nikolić, Luze Sun, Jie Zhang, Florian Tramèr

**Abstract**: Jailbreak attacks bypass the guardrails of large language models to produce harmful outputs. In this paper, we ask whether the model outputs produced by existing jailbreaks are actually useful. For example, when jailbreaking a model to give instructions for building a bomb, does the jailbreak yield good instructions? Since the utility of most unsafe answers (e.g., bomb instructions) is hard to evaluate rigorously, we build new jailbreak evaluation sets with known ground truth answers, by aligning models to refuse questions related to benign and easy-to-evaluate topics (e.g., biology or math). Our evaluation of eight representative jailbreaks across five utility benchmarks reveals a consistent drop in model utility in jailbroken responses, which we term the jailbreak tax. For example, while all jailbreaks we tested bypass guardrails in models aligned to refuse to answer math, this comes at the expense of a drop of up to 92% in accuracy. Overall, our work proposes the jailbreak tax as a new important metric in AI safety, and introduces benchmarks to evaluate existing and future jailbreaks. We make the benchmark available at https://github.com/ethz-spylab/jailbreak-tax

摘要: 越狱攻击绕过大型语言模型的护栏，产生有害的输出。在本文中，我们询问现有越狱产生的模型输出是否真正有用。例如，当越狱模型以给出制造炸弹的指令时，越狱是否会产生良好的指令？由于大多数不安全答案（例如，炸弹指令）很难严格评估，我们通过调整模型来拒绝与良性且易于评估的主题相关的问题，使用已知的地面真相答案来构建新的越狱评估集（例如，生物学或数学）。我们对五个公用事业基准中八个代表性越狱的评估显示，越狱响应（我们将其称为越狱税）中的模型效用持续下降。例如，虽然我们测试的所有越狱都是在拒绝回答数学问题的模型中绕过护栏，但这是以准确性下降高达92%为代价的。总体而言，我们的工作提出将越狱税作为人工智能安全的新的重要指标，并引入了评估现有和未来越狱的基准。我们在https://github.com/ethz-spylab/jailbreak-tax上提供基准测试



## **31. Look Before You Leap: Enhancing Attention and Vigilance Regarding Harmful Content with GuidelineLLM**

三思而后行：通过GuidelineLLM提高对有害内容的关注和警惕 cs.CL

AAAI 2025

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2412.10423v2) [paper-pdf](http://arxiv.org/pdf/2412.10423v2)

**Authors**: Shaoqing Zhang, Zhuosheng Zhang, Kehai Chen, Rongxiang Weng, Muyun Yang, Tiejun Zhao, Min Zhang

**Abstract**: Despite being empowered with alignment mechanisms, large language models (LLMs) are increasingly vulnerable to emerging jailbreak attacks that can compromise their alignment mechanisms. This vulnerability poses significant risks to real-world applications. Existing work faces challenges in both training efficiency and generalization capabilities (i.e., Reinforcement Learning from Human Feedback and Red-Teaming). Developing effective strategies to enable LLMs to resist continuously evolving jailbreak attempts represents a significant challenge. To address this challenge, we propose a novel defensive paradigm called GuidelineLLM, which assists LLMs in recognizing queries that may have harmful content. Before LLMs respond to a query, GuidelineLLM first identifies potential risks associated with the query, summarizes these risks into guideline suggestions, and then feeds these guidelines to the responding LLMs. Importantly, our approach eliminates the necessity for additional safety fine-tuning of the LLMs themselves; only the GuidelineLLM requires fine-tuning. This characteristic enhances the general applicability of GuidelineLLM across various LLMs. Experimental results demonstrate that GuidelineLLM can significantly reduce the attack success rate (ASR) against LLM (an average reduction of 34.17\% ASR) while maintaining the usefulness of LLM in handling benign queries. The code is available at https://github.com/sqzhang-lazy/GuidelineLLM.

摘要: 尽管拥有对齐机制，大型语言模型（LLM）仍越来越容易受到新出现的越狱攻击的影响，这些攻击可能会损害其对齐机制。此漏洞对现实世界的应用程序构成了重大风险。现有工作在培训效率和概括能力方面面临挑战（即，来自人类反馈和红色团队的强化学习）。制定有效的策略以使法学硕士能够抵御不断变化的越狱企图是一项重大挑战。为了应对这一挑战，我们提出了一种名为GuidelineLLM的新型防御范式，它帮助LLM识别可能包含有害内容的查询。在LLM响应查询之前，GuidelineLLM首先识别与查询相关的潜在风险，将这些风险总结为指南建议，然后将这些指南提供给响应的LLM。重要的是，我们的方法消除了对LLM本身进行额外安全微调的必要性;只有GuidelineLLM需要微调。该特征增强了GuidelineLLM在各种LLM中的普遍适用性。实验结果表明，GuidelineLLM可以显着降低针对LLM的攻击成功率（ASB）（平均降低34.17%ASB），同时保持LLM处理良性查询的有用性。该代码可在https://github.com/sqzhang-lazy/GuidelineLLM上获取。



## **32. Using Large Language Models for Template Detection from Security Event Logs**

使用大型语言模型从安全事件收件箱进行模板检测 cs.CR

Accepted for publication in International Journal of Information  Security

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2409.05045v3) [paper-pdf](http://arxiv.org/pdf/2409.05045v3)

**Authors**: Risto Vaarandi, Hayretdin Bahsi

**Abstract**: In modern IT systems and computer networks, real-time and offline event log analysis is a crucial part of cyber security monitoring. In particular, event log analysis techniques are essential for the timely detection of cyber attacks and for assisting security experts with the analysis of past security incidents. The detection of line patterns or templates from unstructured textual event logs has been identified as an important task of event log analysis since detected templates represent event types in the event log and prepare the logs for downstream online or offline security monitoring tasks. During the last two decades, a number of template mining algorithms have been proposed. However, many proposed algorithms rely on traditional data mining techniques, and the usage of Large Language Models (LLMs) has received less attention so far. Also, most approaches that harness LLMs are supervised, and unsupervised LLM-based template mining remains an understudied area. The current paper addresses this research gap and investigates the application of LLMs for unsupervised detection of templates from unstructured security event logs.

摘要: 在现代IT系统和计算机网络中，实时和离线事件日志分析是网络安全监控的重要组成部分。特别是，事件日志分析技术对于及时检测网络攻击和协助安全专家分析过去的安全事件至关重要。从非结构化文本事件日志中检测线模式或模板已被确定为事件日志分析的重要任务，因为检测到的模板代表事件日志中的事件类型，并为下游在线或离线安全监控任务准备日志。在过去的二十年里，人们提出了许多模板挖掘算法。然而，许多提出的算法依赖于传统的数据挖掘技术，并且到目前为止，大型语言模型（LLM）的使用受到的关注较少。此外，大多数利用LLM的方法都是有监督的，而无监督的基于LLM的模板挖掘仍然是一个研究不足的领域。当前论文解决了这一研究空白，并研究了LLM在无监督检测非结构化安全事件日志模板中的应用。



## **33. Benchmarking Practices in LLM-driven Offensive Security: Testbeds, Metrics, and Experiment Design**

法学硕士驱动的攻击性安全中的基准实践：测试床、工作组和实验设计 cs.CR

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10112v1) [paper-pdf](http://arxiv.org/pdf/2504.10112v1)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: Large Language Models (LLMs) have emerged as a powerful approach for driving offensive penetration-testing tooling. This paper analyzes the methodology and benchmarking practices used for evaluating Large Language Model (LLM)-driven attacks, focusing on offensive uses of LLMs in cybersecurity. We review 16 research papers detailing 15 prototypes and their respective testbeds.   We detail our findings and provide actionable recommendations for future research, emphasizing the importance of extending existing testbeds, creating baselines, and including comprehensive metrics and qualitative analysis. We also note the distinction between security research and practice, suggesting that CTF-based challenges may not fully represent real-world penetration testing scenarios.

摘要: 大型语言模型（LLM）已成为驱动攻击性渗透测试工具的强大方法。本文分析了用于评估大型语言模型（LLM）驱动的攻击的方法论和基准实践，重点关注LLM在网络安全中的攻击性使用。我们回顾了16篇研究论文，详细介绍了15个原型及其各自的测试平台。   我们详细介绍了我们的调查结果，并为未来的研究提供了可操作的建议，强调了扩展现有测试平台、创建基线以及包括全面指标和定性分析的重要性。我们还注意到安全研究和实践之间的区别，这表明基于CTF的挑战可能无法完全代表真实世界的渗透测试场景。



## **34. From Vulnerabilities to Remediation: A Systematic Literature Review of LLMs in Code Security**

从漏洞到补救：代码安全领域LLM的系统文献评论 cs.CR

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2412.15004v3) [paper-pdf](http://arxiv.org/pdf/2412.15004v3)

**Authors**: Enna Basic, Alberto Giaretta

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for automating various programming tasks, including security-related ones, such as detecting and fixing vulnerabilities. Despite their promising capabilities, when required to produce or modify pre-existing code, LLMs could introduce vulnerabilities unbeknown to the programmer. When analyzing code, they could miss clear vulnerabilities or signal nonexistent ones. In this Systematic Literature Review (SLR), we aim to investigate both the security benefits and potential drawbacks of using LLMs for a variety of code-related tasks. In particular, first we focus on the types of vulnerabilities that could be introduced by LLMs, when used for producing code. Second, we analyze the capabilities of LLMs to detect and fix vulnerabilities, in any given code, and how the prompting strategy of choice impacts their performance in these two tasks. Last, we provide an in-depth analysis on how data poisoning attacks on LLMs can impact performance in the aforementioned tasks.

摘要: 大型语言模型（LLM）已成为自动化各种编程任务的强大工具，包括与安全相关的任务，例如检测和修复漏洞。尽管LLM的功能很有希望，但当需要生成或修改预先存在的代码时，LLM可能会引入程序员不知道的漏洞。分析代码时，他们可能会错过明显的漏洞或发出不存在的漏洞。在本系统文献评论（SLR）中，我们的目标是研究使用LLM执行各种代码相关任务的安全益处和潜在缺陷。特别是，首先我们关注LLM在用于生成代码时可能引入的漏洞类型。其次，我们分析LLM在任何给定代码中检测和修复漏洞的能力，以及选择的提示策略如何影响其在这两项任务中的性能。最后，我们深入分析了对LLM的数据中毒攻击如何影响上述任务的性能。



## **35. Investigating cybersecurity incidents using large language models in latest-generation wireless networks**

在最新一代无线网络中使用大型语言模型调查网络安全事件 cs.CR

11 pages, 2 figures

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.13196v1) [paper-pdf](http://arxiv.org/pdf/2504.13196v1)

**Authors**: Leonid Legashev, Arthur Zhigalov

**Abstract**: The purpose of research: Detection of cybersecurity incidents and analysis of decision support and assessment of the effectiveness of measures to counter information security threats based on modern generative models. The methods of research: Emulation of signal propagation data in MIMO systems, synthesis of adversarial examples, execution of adversarial attacks on machine learning models, fine tuning of large language models for detecting adversarial attacks, explainability of decisions on detecting cybersecurity incidents based on the prompts technique. Scientific novelty: A binary classification of data poisoning attacks was performed using large language models, and the possibility of using large language models for investigating cybersecurity incidents in the latest generation wireless networks was investigated. The result of research: Fine-tuning of large language models was performed on the prepared data of the emulated wireless network segment. Six large language models were compared for detecting adversarial attacks, and the capabilities of explaining decisions made by a large language model were investigated. The Gemma-7b model showed the best results according to the metrics Precision = 0.89, Recall = 0.89 and F1-Score = 0.89. Based on various explainability prompts, the Gemma-7b model notes inconsistencies in the compromised data under study, performs feature importance analysis and provides various recommendations for mitigating the consequences of adversarial attacks. Large language models integrated with binary classifiers of network threats have significant potential for practical application in the field of cybersecurity incident investigation, decision support and assessing the effectiveness of measures to counter information security threats.

摘要: 研究目的：基于现代生成模型检测网络安全事件、分析决策支持以及评估应对信息安全威胁措施的有效性。研究方法：仿真多输出系统中的信号传播数据、合成对抗性示例、对机器学习模型执行对抗性攻击、微调用于检测对抗性攻击的大型语言模型、基于提示技术检测网络安全事件的决策的可解释性。科学新颖性：使用大型语言模型对数据中毒攻击进行了二进制分类，并研究了使用大型语言模型调查最新一代无线网络中网络安全事件的可能性。研究结果：对模拟无线网络段的准备数据进行了大型语言模型的微调。比较了六种大型语言模型用于检测对抗攻击，并研究了解释大型语言模型所做决策的能力。Gemma-7 b模型显示了最好的结果，根据指标精度= 0.89、召回率= 0.89和F1-Score = 0.89。基于各种可解释性提示，Gemma-7 b模型注意到正在研究的受攻击数据中的不一致性，执行特征重要性分析，并提供各种减轻对抗性攻击后果的建议。与网络威胁二进制分类器集成的大型语言模型在网络安全事件调查、决策支持和评估应对信息安全威胁措施的有效性领域具有巨大的实际应用潜力。



## **36. Do We Really Need Curated Malicious Data for Safety Alignment in Multi-modal Large Language Models?**

在多模态大型语言模型中，我们真的需要精心策划的恶意数据来进行安全对齐吗？ cs.CR

Accepted to CVPR 2025, codes in process

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.10000v1) [paper-pdf](http://arxiv.org/pdf/2504.10000v1)

**Authors**: Yanbo Wang, Jiyang Guan, Jian Liang, Ran He

**Abstract**: Multi-modal large language models (MLLMs) have made significant progress, yet their safety alignment remains limited. Typically, current open-source MLLMs rely on the alignment inherited from their language module to avoid harmful generations. However, the lack of safety measures specifically designed for multi-modal inputs creates an alignment gap, leaving MLLMs vulnerable to vision-domain attacks such as typographic manipulation. Current methods utilize a carefully designed safety dataset to enhance model defense capability, while the specific knowledge or patterns acquired from the high-quality dataset remain unclear. Through comparison experiments, we find that the alignment gap primarily arises from data distribution biases, while image content, response quality, or the contrastive behavior of the dataset makes little contribution to boosting multi-modal safety. To further investigate this and identify the key factors in improving MLLM safety, we propose finetuning MLLMs on a small set of benign instruct-following data with responses replaced by simple, clear rejection sentences. Experiments show that, without the need for labor-intensive collection of high-quality malicious data, model safety can still be significantly improved, as long as a specific fraction of rejection data exists in the finetuning set, indicating the security alignment is not lost but rather obscured during multi-modal pretraining or instruction finetuning. Simply correcting the underlying data bias could narrow the safety gap in the vision domain.

摘要: 多模式大型语言模型（MLLM）已经取得了重大进展，但其安全性一致性仍然有限。通常，当前的开源MLLM依赖于从其语言模块继承的对齐来避免有害的世代。然而，缺乏专门为多模式输入设计的安全措施造成了对齐差距，使MLLM容易受到印刷操纵等视觉域攻击。当前的方法利用精心设计的安全数据集来增强模型防御能力，而从高质量数据集获取的具体知识或模式仍然不清楚。通过比较实验，我们发现对齐差距主要源于数据分布偏差，而图像内容、响应质量或数据集的对比行为对提高多模式安全性贡献不大。为了进一步研究这一点并确定提高MLLM安全性的关键因素，我们建议对一小组良性预算跟踪数据进行微调MLLM，并用简单、明确的拒绝句取代响应。实验表明，在不需要劳动密集型收集高质量恶意数据的情况下，只要微调集中存在特定比例的拒绝数据，模型安全性仍然可以显着提高，这表明安全对齐并没有丢失，而是在多模式预训练或指令微调期间被掩盖。简单地纠正潜在的数据偏见就可以缩小视觉领域的安全差距。



## **37. You've Changed: Detecting Modification of Black-Box Large Language Models**

你已经改变了：检测黑盒大型语言模型的修改 cs.CL

26 pages, 4 figures

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.12335v1) [paper-pdf](http://arxiv.org/pdf/2504.12335v1)

**Authors**: Alden Dima, James Foulds, Shimei Pan, Philip Feldman

**Abstract**: Large Language Models (LLMs) are often provided as a service via an API, making it challenging for developers to detect changes in their behavior. We present an approach to monitor LLMs for changes by comparing the distributions of linguistic and psycholinguistic features of generated text. Our method uses a statistical test to determine whether the distributions of features from two samples of text are equivalent, allowing developers to identify when an LLM has changed. We demonstrate the effectiveness of our approach using five OpenAI completion models and Meta's Llama 3 70B chat model. Our results show that simple text features coupled with a statistical test can distinguish between language models. We also explore the use of our approach to detect prompt injection attacks. Our work enables frequent LLM change monitoring and avoids computationally expensive benchmark evaluations.

摘要: 大型语言模型（LLM）通常通过API作为服务提供，这使得开发人员很难检测其行为的变化。我们提出了一种通过比较生成文本的语言和心理语言特征的分布来监控LLM变化的方法。我们的方法使用统计测试来确定两个文本样本的特征分布是否等效，从而允许开发人员识别LLM何时发生更改。我们使用五个OpenAI完成模型和Meta的Llama 3 70B聊天模型来证明我们方法的有效性。我们的研究结果表明，简单的文本特征加上统计测试可以区分语言模型。我们还探讨了使用我们的方法来检测即时注入攻击。我们的工作使频繁的LLM变化监测，并避免计算昂贵的基准评估。



## **38. StruPhantom: Evolutionary Injection Attacks on Black-Box Tabular Agents Powered by Large Language Models**

StruPhantom：对由大型语言模型支持的黑盒表代理的进化注入攻击 cs.CR

Work in Progress

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.09841v1) [paper-pdf](http://arxiv.org/pdf/2504.09841v1)

**Authors**: Yang Feng, Xudong Pan

**Abstract**: The proliferation of autonomous agents powered by large language models (LLMs) has revolutionized popular business applications dealing with tabular data, i.e., tabular agents. Although LLMs are observed to be vulnerable against prompt injection attacks from external data sources, tabular agents impose strict data formats and predefined rules on the attacker's payload, which are ineffective unless the agent navigates multiple layers of structural data to incorporate the payload. To address the challenge, we present a novel attack termed StruPhantom which specifically targets black-box LLM-powered tabular agents. Our attack designs an evolutionary optimization procedure which continually refines attack payloads via the proposed constrained Monte Carlo Tree Search augmented by an off-topic evaluator. StruPhantom helps systematically explore and exploit the weaknesses of target applications to achieve goal hijacking. Our evaluation validates the effectiveness of StruPhantom across various LLM-based agents, including those on real-world platforms, and attack scenarios. Our attack achieves over 50% higher success rates than baselines in enforcing the application's response to contain phishing links or malicious codes.

摘要: 由大型语言模型（LLM）驱动的自主代理的激增彻底改变了处理表格数据的流行业务应用程序，即片状药剂。尽管LLM被观察到容易受到来自外部数据源的即时注入攻击，但表格代理对攻击者的有效负载施加了严格的数据格式和预定义的规则，除非代理导航多层结构数据以合并有效负载，否则这些规则是无效的。为了应对这一挑战，我们提出了一种名为StruPhantom的新型攻击，该攻击专门针对黑匣子LLM供电的表格代理。我们的攻击设计了一个进化优化过程，该过程通过提出的由非主题评估器增强的约束蒙特卡洛树搜索来不断细化攻击有效负载。StruPhantom帮助系统性地探索和利用目标应用程序的弱点来实现目标劫持。我们的评估验证了StruPhantom在各种基于LLM的代理（包括现实世界平台上的代理）和攻击场景中的有效性。在强制应用程序响应以包含网络钓鱼链接或恶意代码方面，我们的攻击的成功率比基线高出50%以上。



## **39. An Investigation of Large Language Models and Their Vulnerabilities in Spam Detection**

垃圾邮件检测中的大型语言模型及其漏洞研究 cs.CR

10 pages; presented at HotSoS'2025 as a work in progress paper

**SubmitDate**: 2025-04-14    [abs](http://arxiv.org/abs/2504.09776v1) [paper-pdf](http://arxiv.org/pdf/2504.09776v1)

**Authors**: Qiyao Tang, Xiangyang Li

**Abstract**: Spam messages continue to present significant challenges to digital users, cluttering inboxes and posing security risks. Traditional spam detection methods, including rules-based, collaborative, and machine learning approaches, struggle to keep up with the rapidly evolving tactics employed by spammers. This project studies new spam detection systems that leverage Large Language Models (LLMs) fine-tuned with spam datasets. More importantly, we want to understand how LLM-based spam detection systems perform under adversarial attacks that purposefully modify spam emails and data poisoning attacks that exploit the differences between the training data and the massages in detection, to which traditional machine learning models are shown to be vulnerable. This experimentation employs two LLM models of GPT2 and BERT and three spam datasets of Enron, LingSpam, and SMSspamCollection for extensive training and testing tasks. The results show that, while they can function as effective spam filters, the LLM models are susceptible to the adversarial and data poisoning attacks. This research provides very useful insights for future applications of LLM models for information security.

摘要: 垃圾邮件继续给数字用户带来重大挑战，使收件箱变得杂乱并构成安全风险。传统的垃圾邮件检测方法，包括基于规则的、协作的和机器学习的方法，很难跟上垃圾邮件发送者所采用的快速发展的策略。该项目研究新的垃圾邮件检测系统，该系统利用经过垃圾邮件数据集微调的大型语言模型（LLM）。更重要的是，我们想了解基于LLM的垃圾邮件检测系统在有目的地修改垃圾邮件的对抗攻击和利用检测中训练数据和按摩之间差异的数据中毒攻击下如何表现，而传统的机器学习模型被证明是脆弱的。该实验使用两种LLM模型GPT 2和BERT以及三种垃圾邮件数据集Enron、LingSpam和SMSspamCollection来执行广泛的培训和测试任务。结果表明，虽然LLM模型可以充当有效的垃圾邮件过滤器，但它们很容易受到对抗性和数据中毒攻击。这项研究为LLM模型在信息安全方面的未来应用提供了非常有用的见解。



## **40. AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender**

AdaSteer：您的对齐LLM本质上是一个自适应越狱防御者 cs.CR

17 pages, 6 figures, 9 tables

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09466v1) [paper-pdf](http://arxiv.org/pdf/2504.09466v1)

**Authors**: Weixiang Zhao, Jiahe Guo, Yulin Hu, Yang Deng, An Zhang, Xingyu Sui, Xinyang Han, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu

**Abstract**: Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.

摘要: 尽管在安全调整方面做出了广泛的努力，但大型语言模型（LLM）仍然容易受到越狱攻击。激活转向提供了一种无需训练的防御方法，但依赖于固定的转向系数，从而导致次优保护和良性输入的错误拒绝增加。为了解决这个问题，我们提出了AdaSteer，这是一种自适应激活引导方法，可以根据输入特征动态调整模型行为。我们确定了两个关键属性：拒绝定律（R-Law），它表明与拒绝方向相反的越狱输入需要更强的引导，以及有害定律（H-Law），它区分对抗性和良性输入。AdaSteer沿着拒绝方向（RD）和有害方向（HD）引导输入表示，并通过逻辑回归学习自适应系数，确保强大的越狱防御，同时保留良性的输入处理。LLaMA-3.1、Gemma-2和Qwen 2.5的实验表明，AdaSteer在多次越狱攻击中优于基线方法，且对效用的影响最小。我们的结果强调了可解释模型内部要素在LLC中实时、灵活的安全执行方面的潜力。



## **41. CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent**

CheatAgent：通过LLM代理攻击LLM授权的推荐系统 cs.CR

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.13192v1) [paper-pdf](http://arxiv.org/pdf/2504.13192v1)

**Authors**: Liang-bo Ning, Shijie Wang, Wenqi Fan, Qing Li, Xin Xu, Hao Chen, Feiran Huang

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems (RecSys) have brought significant advances in personalized user experience and have attracted considerable attention. Despite the impressive progress, the research question regarding the safety vulnerability of LLM-empowered RecSys still remains largely under-investigated. Given the security and privacy concerns, it is more practical to focus on attacking the black-box RecSys, where attackers can only observe the system's inputs and outputs. However, traditional attack approaches employing reinforcement learning (RL) agents are not effective for attacking LLM-empowered RecSys due to the limited capabilities in processing complex textual inputs, planning, and reasoning. On the other hand, LLMs provide unprecedented opportunities to serve as attack agents to attack RecSys because of their impressive capability in simulating human-like decision-making processes. Therefore, in this paper, we propose a novel attack framework called CheatAgent by harnessing the human-like capabilities of LLMs, where an LLM-based agent is developed to attack LLM-Empowered RecSys. Specifically, our method first identifies the insertion position for maximum impact with minimal input modification. After that, the LLM agent is designed to generate adversarial perturbations to insert at target positions. To further improve the quality of generated perturbations, we utilize the prompt tuning technique to improve attacking strategies via feedback from the victim RecSys iteratively. Extensive experiments across three real-world datasets demonstrate the effectiveness of our proposed attacking method.

摘要: 最近，基于大语言模型（LLM）的推荐系统（RecSys）在个性化用户体验方面带来了显着进步，并引起了相当大的关注。尽管取得了令人印象深刻的进展，但有关LLM授权的RecSys安全漏洞的研究问题仍然基本上没有得到充分的调查。考虑到安全和隐私问题，更实际的做法是专注于攻击黑匣子RecSys，攻击者只能观察系统的输入和输出。然而，由于处理复杂文本输入、规划和推理的能力有限，使用强化学习（RL）代理的传统攻击方法对于攻击LLM授权的RecSys并不有效。另一方面，LLM提供了前所未有的机会作为攻击代理来攻击RecSys，因为它们在模拟类人决策过程方面具有令人印象深刻的能力。因此，在本文中，我们通过利用LLM的类人能力提出了一种名为CheatAgent的新型攻击框架，其中开发了一个基于LLM的代理来攻击LLM授权的RecSys。具体来说，我们的方法首先识别插入位置，以最小的输入修改获得最大影响。之后，LLM代理被设计为生成对抗性扰动以插入目标位置。为了进一步提高生成的扰动的质量，我们利用即时调整技术通过受害者RecSys的反馈迭代改进攻击策略。三个现实世界数据集的广泛实验证明了我们提出的攻击方法的有效性。



## **42. SaRO: Enhancing LLM Safety through Reasoning-based Alignment**

SaRO：通过基于推理的一致提高LLM安全性 cs.CL

**SubmitDate**: 2025-04-13    [abs](http://arxiv.org/abs/2504.09420v1) [paper-pdf](http://arxiv.org/pdf/2504.09420v1)

**Authors**: Yutao Mou, Yuxiao Luo, Shikun Zhang, Wei Ye

**Abstract**: Current safety alignment techniques for large language models (LLMs) face two key challenges: (1) under-generalization, which leaves models vulnerable to novel jailbreak attacks, and (2) over-alignment, which leads to the excessive refusal of benign instructions. Our preliminary investigation reveals semantic overlap between jailbreak/harmful queries and normal prompts in embedding space, suggesting that more effective safety alignment requires a deeper semantic understanding. This motivates us to incorporate safety-policy-driven reasoning into the alignment process. To this end, we propose the Safety-oriented Reasoning Optimization Framework (SaRO), which consists of two stages: (1) Reasoning-style Warmup (RW) that enables LLMs to internalize long-chain reasoning through supervised fine-tuning, and (2) Safety-oriented Reasoning Process Optimization (SRPO) that promotes safety reflection via direct preference optimization (DPO). Extensive experiments demonstrate the superiority of SaRO over traditional alignment methods.

摘要: 当前大型语言模型（LLM）的安全对齐技术面临两个关键挑战：（1）泛化不足，这使得模型容易受到新的越狱攻击，以及（2）过度对齐，这导致过度拒绝良性指令。我们的初步调查揭示了越狱/有害查询和嵌入空间中的正常提示之间的语义重叠，这表明更有效的安全对齐需要更深入的语义理解。这促使我们将安全策略驱动的推理纳入对齐过程。为此，我们提出了面向安全的推理优化框架（SaRO），它包括两个阶段：（1）推理式预热（RW），使LLM能够通过监督微调内化长链推理，以及（2）面向安全的推理过程优化（SRPO），通过直接偏好优化（DPO）促进安全反思。大量实验证明了SaRO相对于传统对齐方法的优越性。



## **43. Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities**

模型篡改攻击能够更严格地评估LLM能力 cs.CR

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2502.05209v2) [paper-pdf](http://arxiv.org/pdf/2502.05209v2)

**Authors**: Zora Che, Stephen Casper, Robert Kirk, Anirudh Satheesh, Stewart Slocum, Lev E McKinney, Rohit Gandikota, Aidan Ewart, Domenic Rosati, Zichu Wu, Zikui Cai, Bilal Chughtai, Yarin Gal, Furong Huang, Dylan Hadfield-Menell

**Abstract**: Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the attack success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.

摘要: 对大型语言模型（LLM）风险和能力的评估越来越多地被纳入人工智能风险管理和治理框架中。目前，大多数风险评估都是通过设计从系统中引发有害行为的输入来进行的。然而，这种方法有两个局限性。首先，投入产出评估无法评估开权模型的现实风险。其次，在任何特定的投入-产出评估期间识别的行为只能下限模型的最坏可能情况的投入-产出行为。作为引发有害行为的补充方法，我们建议使用模型篡改攻击来评估LLM，该攻击允许修改潜在激活或权重。我们使用最先进的技术来消除有害的LLM功能，以对抗一系列5个输入空间和6个模型篡改攻击。除了对这些方法进行比较之外，我们还表明：（1）模型对能力启发攻击的弹性取决于低维鲁棒性子空间;（2）模型篡改攻击的攻击成功率可以根据经验预测并为输入空间攻击的成功提供保守估计;和（3）最先进的取消学习方法可以在16个微调步骤内轻松取消。这些结果共同强调了抑制有害LLM功能的难度，并表明模型篡改攻击比单独的输入空间攻击可以实现更严格的评估。



## **44. Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities**

基于有限样本浓度不等式的LLM文本检测零次统计检验 stat.ML

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2501.02406v3) [paper-pdf](http://arxiv.org/pdf/2501.02406v3)

**Authors**: Tara Radvand, Mojtaba Abdolmaleki, Mohamed Mostagir, Ambuj Tewari

**Abstract**: Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly challenging as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. We answer the following question: Given a piece of text, can we identify whether it was produced by LLM $A$ or $B$ (where $B$ can be a human)? We model LLM-generated text as a sequential stochastic process with complete dependence on history and design zero-shot statistical tests to distinguish between (i) the text generated by two different sets of LLMs $A$ (in-house) and $B$ (non-sanctioned) and also (ii) LLM-generated and human-generated texts. We prove that our tests' type I and type II errors decrease exponentially as text length increases. For designing our tests for a given string, we demonstrate that if the string is generated by the evaluator model $A$, the log-perplexity of the string under $A$ converges to the average entropy of the string under $A$, except with an exponentially small probability in the string length. We also show that if $B$ generates the text, except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. For our experiments: First, we present experiments using open-source LLMs to support our theoretical results, and then we provide experiments in a black-box setting with adversarial attacks. Practically, our work enables guaranteed finding of the origin of harmful or false LLM-generated text, which can be useful for combating misinformation and compliance with emerging AI regulations.

摘要: 验证内容的出处对于许多组织的功能至关重要，例如，教育机构、社交媒体平台、公司等。随着大型语言模型（LLM）生成的文本与人类生成的内容几乎无法区分，这个问题变得越来越具有挑战性。此外，许多机构利用内部LLM，并希望确保外部未经批准的LLM不会在机构内制作内容。我们回答以下问题：给定一段文本，我们可以识别它是由LLM $A$还是$B$（其中$B$可以是人类）生成的吗？我们将LLM生成的文本建模为一个顺序随机过程，完全依赖于历史，并设计零次统计测试来区分（i）由两组不同的LLM $A$（内部）和$B$（非认可）生成的文本，以及（ii）LLM生成的文本和人类生成的文本。我们证明了我们的测试的类型I和类型II错误随着文本长度的增加而呈指数级下降。为了设计我们的测试对于一个给定的字符串，我们证明，如果字符串是由评估模型$A$，下$A$的字符串的对数困惑收敛到下$A$的字符串的平均熵，除了在字符串长度的指数小的概率。我们还表明，如果$B$生成的文本，除了一个指数小概率的字符串长度，下$A$的字符串的对数困惑收敛到平均交叉熵的$B$和$A$。对于我们的实验：首先，我们使用开源LLM进行实验来支持我们的理论结果，然后我们在具有对抗性攻击的黑匣子环境中提供实验。实际上，我们的工作能够保证找到LLM生成的有害或虚假文本的来源，这对于打击错误信息和遵守新出现的人工智能法规非常有用。



## **45. Feature-Aware Malicious Output Detection and Mitigation**

具有攻击意识的恶意输出检测和缓解 cs.CL

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2504.09191v1) [paper-pdf](http://arxiv.org/pdf/2504.09191v1)

**Authors**: Weilong Dong, Peiguang Li, Yu Tian, Xinyi Zeng, Fengdi Li, Sirui Wang

**Abstract**: The rapid advancement of large language models (LLMs) has brought significant benefits to various domains while introducing substantial risks. Despite being fine-tuned through reinforcement learning, LLMs lack the capability to discern malicious content, limiting their defense against jailbreak. To address these safety concerns, we propose a feature-aware method for harmful response rejection (FMM), which detects the presence of malicious features within the model's feature space and adaptively adjusts the model's rejection mechanism. By employing a simple discriminator, we detect potential malicious traits during the decoding phase. Upon detecting features indicative of toxic tokens, FMM regenerates the current token. By employing activation patching, an additional rejection vector is incorporated during the subsequent token generation, steering the model towards a refusal response. Experimental results demonstrate the effectiveness of our approach across multiple language models and diverse attack techniques, while crucially maintaining the models' standard generation capabilities.

摘要: 大型语言模型（LLM）的快速发展为各个领域带来了巨大的好处，同时也带来了巨大的风险。尽管通过强化学习进行了微调，但LLM缺乏识别恶意内容的能力，限制了他们对越狱的防御。为了解决这些安全问题，我们提出了一种用于有害响应拒绝（FMM）的特征感知方法，该方法检测模型特征空间内是否存在恶意特征，并自适应地调整模型的拒绝机制。通过使用简单的收件箱，我们在解码阶段检测潜在的恶意特征。检测到指示有毒令牌的特征后，FMM会重新生成当前令牌。通过使用激活补丁，在后续令牌生成期间合并额外的拒绝载体，引导模型转向拒绝响应。实验结果证明了我们的方法在多种语言模型和多种攻击技术中的有效性，同时至关重要地保持了模型的标准生成能力。



## **46. Privacy Preservation in Gen AI Applications**

世代人工智能应用程序中的隐私保护 cs.CR

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2504.09095v1) [paper-pdf](http://arxiv.org/pdf/2504.09095v1)

**Authors**: Swetha S, Ram Sundhar K Shaju, Rakshana M, Ganesh R, Balavedhaa S, Thiruvaazhi U

**Abstract**: The ability of machines to comprehend and produce language that is similar to that of humans has revolutionized sectors like customer service, healthcare, and finance thanks to the quick advances in Natural Language Processing (NLP), which are fueled by Generative Artificial Intelligence (AI) and Large Language Models (LLMs). However, because LLMs trained on large datasets may unintentionally absorb and reveal Personally Identifiable Information (PII) from user interactions, these capabilities also raise serious privacy concerns. Deep neural networks' intricacy makes it difficult to track down or stop the inadvertent storing and release of private information, which raises serious concerns about the privacy and security of AI-driven data. This study tackles these issues by detecting Generative AI weaknesses through attacks such as data extraction, model inversion, and membership inference. A privacy-preserving Generative AI application that is resistant to these assaults is then developed. It ensures privacy without sacrificing functionality by using methods to identify, alter, or remove PII before to dealing with LLMs. In order to determine how well cloud platforms like Microsoft Azure, Google Cloud, and AWS provide privacy tools for protecting AI applications, the study also examines these technologies. In the end, this study offers a fundamental privacy paradigm for generative AI systems, focusing on data security and moral AI implementation, and opening the door to a more secure and conscientious use of these tools.

摘要: 由于自然语言处理（NLP）的快速发展，机器理解和产生与人类类似的语言的能力彻底改变了客户服务、医疗保健和金融等领域，这要归功于生成人工智能（AI）和大型语言模型（LLM）。然而，由于在大型数据集上训练的LLM可能会无意中吸收和泄露来自用户交互的个人可识别信息（PRI），因此这些功能也会引发严重的隐私问题。深度神经网络的复杂性使得很难追踪或阻止私人信息的无意存储和发布，这引发了人们对人工智能驱动数据的隐私和安全性的严重担忧。这项研究通过数据提取、模型倒置和隶属度推断等攻击来检测生成性人工智能的弱点来解决这些问题。然后开发出一个能够抵抗这些攻击的保护隐私的生成人工智能应用程序。它通过在处理LLM之前使用识别、更改或删除PRI的方法来确保隐私，而不会牺牲功能。为了确定Microsoft Azure、Google Cloud和AWS等云平台为保护人工智能应用程序提供的隐私工具的效果如何，该研究还研究了这些技术。最后，这项研究为生成性人工智能系统提供了一个基本的隐私范式，重点关注数据安全和道德人工智能实施，并为更安全、更认真地使用这些工具打开了大门。



## **47. Detecting Instruction Fine-tuning Attack on Language Models with Influence Function**

利用影响函数检测语言模型的指令微调攻击 cs.LG

**SubmitDate**: 2025-04-12    [abs](http://arxiv.org/abs/2504.09026v1) [paper-pdf](http://arxiv.org/pdf/2504.09026v1)

**Authors**: Jiawei Li

**Abstract**: Instruction fine-tuning attacks pose a significant threat to large language models (LLMs) by subtly embedding poisoned data in fine-tuning datasets, which can trigger harmful or unintended responses across a range of tasks. This undermines model alignment and poses security risks in real-world deployment. In this work, we present a simple and effective approach to detect and mitigate such attacks using influence functions, a classical statistical tool adapted for machine learning interpretation. Traditionally, the high computational costs of influence functions have limited their application to large models and datasets. The recent Eigenvalue-Corrected Kronecker-Factored Approximate Curvature (EK-FAC) approximation method enables efficient influence score computation, making it feasible for large-scale analysis.   We are the first to apply influence functions for detecting language model instruction fine-tuning attacks on large-scale datasets, as both the instruction fine-tuning attack on language models and the influence calculation approximation technique are relatively new. Our large-scale empirical evaluation of influence functions on 50,000 fine-tuning examples and 32 tasks reveals a strong association between influence scores and sentiment. Building on this, we introduce a novel sentiment transformation combined with influence functions to detect and remove critical poisons -- poisoned data points that skew model predictions. Removing these poisons (only 1% of total data) recovers model performance to near-clean levels, demonstrating the effectiveness and efficiency of our approach. Artifact is available at https://github.com/lijiawei20161002/Poison-Detection.   WARNING: This paper contains offensive data examples.

摘要: 指令微调攻击通过巧妙地将有毒数据嵌入微调数据集中，对大型语言模型（LLM）构成重大威胁，这可能会在一系列任务中触发有害或意外的响应。这会破坏模型一致性，并在现实世界部署中带来安全风险。在这项工作中，我们提出了一种简单有效的方法来使用影响函数来检测和减轻此类攻击，影响函数是一种适合机器学习解释的经典统计工具。传统上，影响函数的高计算成本限制了其对大型模型和数据集的应用。最近的特征值修正克罗内克因子逼近曲线（EK-FAC）逼近方法能够实现高效的影响分数计算，使其适合大规模分析。   我们是第一个应用影响函数来检测对大规模数据集的语言模型指令微调攻击的人，因为对语言模型的指令微调攻击和影响计算逼近技术都是相对较新的。我们对50，000个微调示例和32个任务的影响力函数进行了大规模实证评估，揭示了影响力分数和情绪之间存在很强的关联。在此基础上，我们引入了一种新颖的情感转换，并结合影响函数来检测和删除关键毒药--扭曲模型预测的有毒数据点。删除这些毒物（仅占总数据的1%）可将模型性能恢复到接近清洁的水平，证明了我们方法的有效性和效率。收件箱可在https://github.com/lijiawei20161002/Poison-Detection上获取。   警告：本文包含攻击性数据示例。



## **48. Robust Steganography from Large Language Models**

来自大型语言模型的稳健隐写术 cs.CR

36 pages, 9 figures

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08977v1) [paper-pdf](http://arxiv.org/pdf/2504.08977v1)

**Authors**: Neil Perry, Sanket Gupte, Nishant Pitta, Lior Rotem

**Abstract**: Recent steganographic schemes, starting with Meteor (CCS'21), rely on leveraging large language models (LLMs) to resolve a historically-challenging task of disguising covert communication as ``innocent-looking'' natural-language communication. However, existing methods are vulnerable to ``re-randomization attacks,'' where slight changes to the communicated text, that might go unnoticed, completely destroy any hidden message. This is also a vulnerability in more traditional encryption-based stegosystems, where adversaries can modify the randomness of an encryption scheme to destroy the hidden message while preserving an acceptable covertext to ordinary users. In this work, we study the problem of robust steganography. We introduce formal definitions of weak and strong robust LLM-based steganography, corresponding to two threat models in which natural language serves as a covertext channel resistant to realistic re-randomization attacks. We then propose two constructions satisfying these notions. We design and implement our steganographic schemes that embed arbitrary secret messages into natural language text generated by LLMs, ensuring recoverability even under adversarial paraphrasing and rewording attacks. To support further research and real-world deployment, we release our implementation and datasets for public use.

摘要: 最近的隐写计划，从Meteor（CCS ' 21）开始，依靠利用大型语言模型（LLM）来解决一项具有历史挑战性的任务，即将秘密通信伪装成“看起来无辜”的自然语言通信。然而，现有的方法很容易受到“重新随机化攻击”，即对所传达的文本的轻微变化（可能不被注意到）会完全破坏任何隐藏的信息。这也是更传统的基于加密的隐写系统中的一个漏洞，其中对手可以修改加密方案的随机性以破坏隐藏消息，同时为普通用户保留可接受的封面文本。在这项工作中，我们研究了稳健隐写术的问题。我们介绍了正式定义的弱和强鲁棒的基于LLM的隐写术，对应于两个威胁模型，其中自然语言作为一个covertext信道抵抗现实的重新随机化攻击。然后，我们提出了两个建设满足这些概念。我们设计并实现了隐写方案，将任意秘密消息嵌入到LLM生成的自然语言文本中，即使在对抗性释义和改写攻击下也能确保可恢复性。为了支持进一步的研究和实际部署，我们发布了我们的实现和数据集供公众使用。



## **49. Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups**

穿越兔子洞：LLM生成的针对心理健康群体的攻击叙事中的紧急偏见 cs.CL

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.06160v3) [paper-pdf](http://arxiv.org/pdf/2504.06160v3)

**Authors**: Rijul Magu, Arka Dutta, Sean Kim, Ashiqur R. KhudaBukhsh, Munmun De Choudhury

**Abstract**: Large Language Models (LLMs) have been shown to demonstrate imbalanced biases against certain groups. However, the study of unprovoked targeted attacks by LLMs towards at-risk populations remains underexplored. Our paper presents three novel contributions: (1) the explicit evaluation of LLM-generated attacks on highly vulnerable mental health groups; (2) a network-based framework to study the propagation of relative biases; and (3) an assessment of the relative degree of stigmatization that emerges from these attacks. Our analysis of a recently released large-scale bias audit dataset reveals that mental health entities occupy central positions within attack narrative networks, as revealed by a significantly higher mean centrality of closeness (p-value = 4.06e-10) and dense clustering (Gini coefficient = 0.7). Drawing from sociological foundations of stigmatization theory, our stigmatization analysis indicates increased labeling components for mental health disorder-related targets relative to initial targets in generation chains. Taken together, these insights shed light on the structural predilections of large language models to heighten harmful discourse and highlight the need for suitable approaches for mitigation.

摘要: 事实证明，大型语言模型（LLM）对某些群体表现出不平衡的偏见。然而，关于LLM对高危人群进行无端有针对性攻击的研究仍然没有得到充分的研究。我们的论文提出了三个新颖的贡献：（1）对LLM产生的对高度弱势心理健康群体的攻击的明确评估;（2）基于网络的框架来研究相对偏见的传播;（3）对这些攻击中出现的耻辱的相对程度的评估。我们对最近发布的大规模偏见审计数据集的分析表明，心理健康实体在攻击叙事网络中占据了中心位置，这一点表现为密切度（p值= 4.06e-10）和密集聚集度（基尼系数= 0.7）的平均中心性显着更高。根据污名化理论的社会学基础，我们的污名化分析表明，相对于代际链中的初始目标，与心理健康疾病相关目标的标签成分有所增加。总而言之，这些见解揭示了大型语言模型加剧有害话语的结构偏好，并强调了适当的缓解方法的必要性。



## **50. MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits**

HCP安全审计：具有模型上下文协议的LLM允许重大安全漏洞 cs.CR

27 pages, 21 figures, and 2 Tables. Cleans up the TeX source

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.03767v2) [paper-pdf](http://arxiv.org/pdf/2504.03767v2)

**Authors**: Brandon Radosevich, John Halloran

**Abstract**: To reduce development overhead and enable seamless integration between potential components comprising any given generative AI application, the Model Context Protocol (MCP) (Anthropic, 2024) has recently been released and subsequently widely adopted. The MCP is an open protocol that standardizes API calls to large language models (LLMs), data sources, and agentic tools. By connecting multiple MCP servers, each defined with a set of tools, resources, and prompts, users are able to define automated workflows fully driven by LLMs. However, we show that the current MCP design carries a wide range of security risks for end users. In particular, we demonstrate that industry-leading LLMs may be coerced into using MCP tools to compromise an AI developer's system through various attacks, such as malicious code execution, remote access control, and credential theft. To proactively mitigate these and related attacks, we introduce a safety auditing tool, MCPSafetyScanner, the first agentic tool to assess the security of an arbitrary MCP server. MCPScanner uses several agents to (a) automatically determine adversarial samples given an MCP server's tools and resources; (b) search for related vulnerabilities and remediations based on those samples; and (c) generate a security report detailing all findings. Our work highlights serious security issues with general-purpose agentic workflows while also providing a proactive tool to audit MCP server safety and address detected vulnerabilities before deployment.   The described MCP server auditing tool, MCPSafetyScanner, is freely available at: https://github.com/johnhalloran321/mcpSafetyScanner

摘要: 为了减少开发费用并实现构成任何给定生成式人工智能应用程序的潜在组件之间的无缝集成，模型上下文协议（HCP）（Anthropic，2024）最近发布并随后广泛采用。HCP是一种开放协议，可同步化对大型语言模型（LLM）、数据源和代理工具的API调用。通过连接多个HCP服务器（每个服务器都定义了一组工具、资源和提示），用户能够定义完全由LLM驱动的自动化工作流程。然而，我们表明当前的LCP设计对最终用户来说存在广泛的安全风险。特别是，我们证明了行业领先的LLM可能会被迫使用LCP工具通过各种攻击（例如恶意代码执行、远程访问控制和凭证盗窃）来危害人工智能开发人员的系统。为了主动缓解这些攻击和相关攻击，我们引入了安全审计工具MCPSafetyScanner，这是第一个评估任意LCP服务器安全性的代理工具。MCPScanner使用多个代理来（a）在给定HCP服务器的工具和资源的情况下自动确定对抗样本;（b）根据这些样本搜索相关漏洞和补救措施;以及（c）生成详细说明所有发现结果的安全报告。我们的工作强调了通用代理工作流程的严重安全问题，同时还提供了一种主动工具来审计LCP服务器的安全性并在部署之前解决检测到的漏洞。   所描述的LCP服务器审计工具MCPSafetyScanner可在以下网址免费获取：https://github.com/johnhalloran321/mcpSafetyScanner



