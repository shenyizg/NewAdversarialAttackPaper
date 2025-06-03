# Latest Large Language Model Attack Papers
**update at 2025-06-03 09:49:02**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Large Language Model-Supported Threat Modeling Framework for Transportation Cyber-Physical Systems**

运输网络物理系统支持的大语言模型威胁建模框架 cs.CR

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2506.00831v1) [paper-pdf](http://arxiv.org/pdf/2506.00831v1)

**Authors**: M Sabbir Salek, Mashrur Chowdhury, Muhaimin Bin Munir, Yuchen Cai, Mohammad Imtiaz Hasan, Jean-Michel Tine, Latifur Khan, Mizanur Rahman

**Abstract**: Modern transportation systems rely on cyber-physical systems (CPS), where cyber systems interact seamlessly with physical systems like transportation-related sensors and actuators to enhance safety, mobility, and energy efficiency. However, growing automation and connectivity increase exposure to cyber vulnerabilities. Existing threat modeling frameworks for transportation CPS are often limited in scope, resource-intensive, and dependent on significant cybersecurity expertise. To address these gaps, we present TraCR-TMF (Transportation Cybersecurity and Resiliency Threat Modeling Framework), a large language model (LLM)-based framework that minimizes expert intervention. TraCR-TMF identifies threats, potential attack techniques, and corresponding countermeasures by leveraging the MITRE ATT&CK matrix through three LLM-based approaches: (i) a retrieval-augmented generation (RAG) method requiring no expert input, (ii) an in-context learning approach requiring low expert input, and (iii) a supervised fine-tuning method requiring moderate expert input. TraCR-TMF also maps attack paths to critical assets by analyzing vulnerabilities using a customized LLM. The framework was evaluated in two scenarios. First, it identified relevant attack techniques across transportation CPS applications, with 90% precision as validated by experts. Second, using a fine-tuned LLM, it successfully predicted multiple exploitations including lateral movement, data exfiltration, and ransomware-related encryption that occurred during a major real-world cyberattack incident. These results demonstrate TraCR-TMF's effectiveness in CPS threat modeling, its reduced reliance on cybersecurity expertise, and its adaptability across CPS domains.

摘要: 现代交通系统依赖于网络物理系统（CPS），其中网络系统与交通相关的传感器和致动器等物理系统无缝交互，以提高安全性、移动性和能源效率。然而，自动化和连接性的不断发展增加了网络漏洞的风险。现有的交通CPS威胁建模框架通常范围有限、资源密集型，并且依赖于重要的网络安全专业知识。为了解决这些差距，我们提出了TraCR-SYS（交通网络安全和弹性威胁建模框架），这是一个基于大型语言模型（LLM）的框架，可以最大限度地减少专家干预。TraCR-SYS通过三种基于LLM的方法利用MITRE ATA & CK矩阵来识别威胁、潜在的攻击技术和相应的对策：（i）不需要专家输入的检索增强生成（RAG）方法，（ii）需要低专家输入的上下文学习方法，和（iii）需要适度专家输入的监督微调方法。TraCR-SYS还通过使用自定义的LLM分析漏洞来将攻击路径映射到关键资产。该框架在两种情况下进行了评估。首先，它识别了运输CPS应用程序中的相关攻击技术，经专家验证的准确率为90%。其次，它使用经过微调的LLM，成功预测了现实世界重大网络攻击事件期间发生的多次利用，包括横向移动、数据泄露和勒索软件相关加密。这些结果证明了TraCR-SYS在CPS威胁建模方面的有效性、减少了对网络安全专业知识的依赖以及其跨CPS域的适应性。



## **2. Jailbreak-R1: Exploring the Jailbreak Capabilities of LLMs via Reinforcement Learning**

越狱-R1：通过强化学习探索LLM的越狱能力 cs.AI

21 pages, 8 figures

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2506.00782v1) [paper-pdf](http://arxiv.org/pdf/2506.00782v1)

**Authors**: Weiyang Guo, Zesheng Shi, Zhuo Li, Yequan Wang, Xuebo Liu, Wenya Wang, Fangming Liu, Min Zhang, Jing Li

**Abstract**: As large language models (LLMs) grow in power and influence, ensuring their safety and preventing harmful output becomes critical. Automated red teaming serves as a tool to detect security vulnerabilities in LLMs without manual labor. However, most existing methods struggle to balance the effectiveness and diversity of red-team generated attack prompts. To address this challenge, we propose \ourapproach, a novel automated red teaming training framework that utilizes reinforcement learning to explore and generate more effective attack prompts while balancing their diversity. Specifically, it consists of three training stages: (1) Cold Start: The red team model is supervised and fine-tuned on a jailbreak dataset obtained through imitation learning. (2) Warm-up Exploration: The model is trained in jailbreak instruction following and exploration, using diversity and consistency as reward signals. (3) Enhanced Jailbreak: Progressive jailbreak rewards are introduced to gradually enhance the jailbreak performance of the red-team model. Extensive experiments on a variety of LLMs show that \ourapproach effectively balances the diversity and effectiveness of jailbreak prompts compared to existing methods. Our work significantly improves the efficiency of red team exploration and provides a new perspective on automated red teaming.

摘要: 随着大型语言模型（LLM）的力量和影响力不断增强，确保其安全性和防止有害输出变得至关重要。自动红色分组可以作为一种无需手工即可检测LLM安全漏洞的工具。然而，大多数现有方法都难以平衡红队生成的攻击提示的有效性和多样性。为了应对这一挑战，我们提出了\ourapproach，这是一种新型的自动化红色团队训练框架，它利用强化学习来探索和生成更有效的攻击提示，同时平衡其多样性。具体来说，它由三个训练阶段组成：（1）冷启动：红队模型在通过模仿学习获得的越狱数据集上进行监督和微调。(2)热身探索：该模型在越狱指导跟踪和探索中进行训练，使用多样性和一致性作为奖励信号。(3)强化越狱：引入渐进式越狱奖励，逐步提升红队模式的越狱表现。对各种LLM的广泛实验表明，与现有方法相比，我们的方法有效地平衡了越狱提示的多样性和有效性。我们的工作显着提高了红色团队探索的效率，并为自动化红色团队合作提供了新的视角。



## **3. CoP: Agentic Red-teaming for Large Language Models using Composition of Principles**

CoP：使用原则组合的大型语言模型的大型红色团队 cs.AI

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2506.00781v1) [paper-pdf](http://arxiv.org/pdf/2506.00781v1)

**Authors**: Chen Xiong, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Recent advances in Large Language Models (LLMs) have spurred transformative applications in various domains, ranging from open-source to proprietary LLMs. However, jailbreak attacks, which aim to break safety alignment and user compliance by tricking the target LLMs into answering harmful and risky responses, are becoming an urgent concern. The practice of red-teaming for LLMs is to proactively explore potential risks and error-prone instances before the release of frontier AI technology. This paper proposes an agentic workflow to automate and scale the red-teaming process of LLMs through the Composition-of-Principles (CoP) framework, where human users provide a set of red-teaming principles as instructions to an AI agent to automatically orchestrate effective red-teaming strategies and generate jailbreak prompts. Distinct from existing red-teaming methods, our CoP framework provides a unified and extensible framework to encompass and orchestrate human-provided red-teaming principles to enable the automated discovery of new red-teaming strategies. When tested against leading LLMs, CoP reveals unprecedented safety risks by finding novel jailbreak prompts and improving the best-known single-turn attack success rate by up to 19.0 times.

摘要: 大型语言模型（LLM）的最新进展激发了各个领域的变革性应用程序，从开源到专有LLM。然而，越狱攻击的目的是通过诱骗目标LLM回答有害和危险的响应来打破安全一致和用户合规性，正在成为一个紧迫的问题。LLM的红色团队实践是在前沿人工智能技术发布之前主动探索潜在风险和容易出错的实例。本文提出了一种代理工作流程，通过构成原则（CoP）框架自动化和扩展LLM的红色团队流程，其中人类用户提供一组红色团队原则作为指令，向人工智能代理自动协调有效的红色团队策略并生成越狱提示。与现有的红色团队方法不同，我们的CoP框架提供了一个统一且可扩展的框架，以涵盖和编排人类提供的红色团队原则，以实现新的红色团队策略的自动发现。当针对领先的LLM进行测试时，CoP发现了新颖的越狱提示并将最著名的单回合攻击成功率提高了19.0倍，从而揭示了前所未有的安全风险。



## **4. Underestimated Privacy Risks for Minority Populations in Large Language Model Unlearning**

在大型语言模型学习中，少数群体的隐私风险被低估 cs.LG

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2412.08559v3) [paper-pdf](http://arxiv.org/pdf/2412.08559v3)

**Authors**: Rongzhe Wei, Mufei Li, Mohsen Ghassemi, Eleonora Kreačić, Yifan Li, Xiang Yue, Bo Li, Vamsi K. Potluru, Pan Li, Eli Chien

**Abstract**: Large Language Models (LLMs) embed sensitive, human-generated data, prompting the need for unlearning methods. Although certified unlearning offers strong privacy guarantees, its restrictive assumptions make it unsuitable for LLMs, giving rise to various heuristic approaches typically assessed through empirical evaluations. These standard evaluations randomly select data for removal, apply unlearning techniques, and use membership inference attacks (MIAs) to compare unlearned models against models retrained without the removed data. However, to ensure robust privacy protections for every data point, it is essential to account for scenarios in which certain data subsets face elevated risks. Prior research suggests that outliers, particularly including data tied to minority groups, often exhibit higher memorization propensity which indicates they may be more difficult to unlearn. Building on these insights, we introduce a complementary, minority-aware evaluation framework to highlight blind spots in existing frameworks. We substantiate our findings with carefully designed experiments, using canaries with personally identifiable information (PII) to represent these minority subsets and demonstrate that they suffer at least 20% higher privacy leakage across various unlearning methods, MIAs, datasets, and LLM scales. Our proposed minority-aware evaluation framework marks an essential step toward more equitable and comprehensive assessments of LLM unlearning efficacy.

摘要: 大型语言模型（LLM）嵌入了敏感的、人类生成的数据，这促使人们需要去学习方法。虽然认证的非学习提供了强大的隐私保证，但其限制性假设使其不适合LLM，从而产生了通常通过经验评估评估的各种启发式方法。这些标准评估随机选择要删除的数据，应用非学习技术，并使用隶属度推理攻击（MIA）来比较未学习的模型与在没有删除数据的情况下重新训练的模型。然而，为了确保每个数据点都有强大的隐私保护，必须考虑某些数据子集面临高风险的情况。之前的研究表明，异常值，特别是包括与少数群体相关的数据，通常表现出更高的记忆倾向，这表明他们可能更难忘记。在这些见解的基础上，我们引入了一个补充的、少数群体意识的评估框架，以突出现有框架中的盲点。我们通过精心设计的实验证实了我们的发现，使用具有个人可识别信息（PRI）的金丝雀来代表这些少数族裔子集，并证明它们在各种取消学习方法、MIA、数据集和LLM量表中遭受的隐私泄露至少高出20%。我们提出的少数族裔意识评估框架标志着朝着更公平、全面的LLM遗忘功效评估迈出了重要的一步。



## **5. Tokens for Learning, Tokens for Unlearning: Mitigating Membership Inference Attacks in Large Language Models via Dual-Purpose Training**

用于学习的代币，用于取消学习的代币：通过双重用途培训减轻大型语言模型中的成员推断攻击 cs.LG

ACL'25 (Findings)

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2502.19726v2) [paper-pdf](http://arxiv.org/pdf/2502.19726v2)

**Authors**: Toan Tran, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) have become the backbone of modern natural language processing but pose privacy concerns about leaking sensitive training data. Membership inference attacks (MIAs), which aim to infer whether a sample is included in a model's training dataset, can serve as a foundation for broader privacy threats. Existing defenses designed for traditional classification models do not account for the sequential nature of text data. As a result, they either require significant computational resources or fail to effectively mitigate privacy risks in LLMs. In this work, we propose \methodname, a lightweight yet effective empirical privacy defense for protecting training data of language models by leveraging token-specific characteristics. By analyzing token dynamics during training, we propose a token selection strategy that categorizes tokens into hard tokens for learning and memorized tokens for unlearning. Subsequently, our training-phase defense optimizes a novel dual-purpose token-level loss to achieve a Pareto-optimal balance between utility and privacy. Extensive experiments demonstrate that our approach not only provides strong protection against MIAs but also improves language modeling performance by around 10\% across various LLM architectures and datasets compared to the baselines.

摘要: 大型语言模型（LLM）已经成为现代自然语言处理的支柱，但也带来了泄露敏感训练数据的隐私问题。成员推断攻击（MIA）旨在推断样本是否包含在模型的训练数据集中，可以作为更广泛的隐私威胁的基础。为传统分类模型设计的现有防御措施没有考虑到文本数据的连续性。因此，它们要么需要大量的计算资源，要么无法有效地减轻LLM中的隐私风险。在这项工作中，我们提出了\MethodName，这是一种轻量级但有效的经验隐私防御，用于通过利用代币特定的特征来保护语言模型的训练数据。通过分析训练期间的令牌动态，我们提出了一种令牌选择策略，将令牌分为用于学习的硬令牌和用于取消学习的记忆令牌。随后，我们的训练阶段防御优化了一种新颖的双重用途代币级损失，以在效用和隐私之间实现帕累托最优平衡。大量实验表明，我们的方法不仅提供了针对MIA的强有力保护，而且与基线相比，还将各种LLM架构和数据集的语言建模性能提高了约10%。



## **6. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

免疫：通过推理时间对齐提高多模式LLM中越狱的安全性 cs.CR

Accepted to CVPR 2025

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2411.18688v4) [paper-pdf](http://arxiv.org/pdf/2411.18688v4)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Alvaro Velasquez, Ahmad Beirami, Furong Huang, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks. In this work, we first highlight an important safety gap to describe that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model through controlled decoding to defend against jailbreak attacks. Additionally, we provide a mathematical characterization of Immune, offering insights on why it improves safety against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.

摘要: 随着多模式大型语言模型（MLLM）用于视觉推理任务的广泛部署，提高其安全性变得至关重要。最近的研究表明，尽管训练时安全一致，但这些模型仍然容易受到越狱攻击。在这项工作中，我们首先强调了一个重要的安全差距，以描述仅通过安全培训实现的对准可能不足以对抗越狱袭击。为了解决这个漏洞，我们提出了Immune，这是一种推理时防御框架，通过受控解码利用安全奖励模型来抵御越狱攻击。此外，我们还提供了Immune的数学描述，并深入了解它为何可以提高越狱安全性。使用最新的MLLM对各种越狱基准进行了广泛评估，结果表明Immune有效地增强了模型的安全性，同时保留了模型的原始功能。例如，针对LLaVA-1.6的基于文本的越狱攻击，与基本MLLM和最先进的防御策略相比，Immune将攻击成功率分别降低了57.82%和16.78%。



## **7. Security Concerns for Large Language Models: A Survey**

大型语言模型的安全性问题综述 cs.CR

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2505.18889v2) [paper-pdf](http://arxiv.org/pdf/2505.18889v2)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as GPT-4 and its recent iterations, Google's Gemini, Anthropic's Claude 3 models, and xAI's Grok have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. In this survey, we provide a comprehensive overview of the emerging security concerns around LLMs, categorizing threats into prompt injection and jailbreaking, adversarial attacks such as input perturbations and data poisoning, misuse by malicious actors for purposes such as generating disinformation, phishing emails, and malware, and worrisome risks inherent in autonomous LLM agents. A significant focus has been recently placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.

摘要: GPT-4及其最近的迭代、Google的Gemini、Anthropic的Claude 3模型和xAI的Grok等大型语言模型（LLM）引发了自然语言处理领域的一场革命，但它们的功能也引入了新的安全漏洞。在本调查中，我们全面概述了围绕LLM的新安全问题，将威胁分为即时注入和越狱、输入干扰和数据中毒等对抗性攻击、恶意行为者出于生成虚假信息、网络钓鱼电子邮件和恶意软件等目的的滥用以及自主LLM代理固有的令人担忧的风险。最近人们对后者给予了极大的关注，探索目标失调、紧急欺骗、自我保护本能，以及LLM制定和追求隐蔽、失调目标的潜力，这种行为被称为阴谋，甚至可能通过安全培训持续存在。我们总结了2022年至2025年期间最近的学术和工业研究，这些研究揭示了每种威胁，分析了拟议的防御措施及其局限性，并确定了保护基于LLM的应用程序方面的公开挑战。最后，我们强调了推进强大的多层安全策略以确保LLM安全且有益的重要性。



## **8. SafeTuneBed: A Toolkit for Benchmarking LLM Safety Alignment in Fine-Tuning**

SafeTuneBed：用于在微调中对LLM安全一致进行基准测试的工具包 cs.LG

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00676v1) [paper-pdf](http://arxiv.org/pdf/2506.00676v1)

**Authors**: Saad Hossain, Samanvay Vajpayee, Sirisha Rambhatla

**Abstract**: As large language models (LLMs) become ubiquitous, parameter-efficient fine-tuning methods and safety-first defenses have proliferated rapidly. However, the number of approaches and their recent increase have resulted in diverse evaluations-varied datasets, metrics, and inconsistent threat settings-making it difficult to fairly compare safety, utility, and robustness across methods. To address this, we introduce SafeTuneBed, a benchmark and toolkit unifying fine-tuning and defense evaluation. SafeTuneBed (i) curates a diverse repository of multiple fine-tuning datasets spanning sentiment analysis, question-answering, multi-step reasoning, and open-ended instruction tasks, and allows for the generation of harmful-variant splits; (ii) enables integration of state-of-the-art defenses, including alignment-stage immunization, in-training safeguards, and post-tuning repair; and (iii) provides evaluators for safety (attack success rate, refusal consistency) and utility. Built on Python-first, dataclass-driven configs and plugins, SafeTuneBed requires minimal additional code to specify any fine-tuning regime, defense method, and metric suite, while ensuring end-to-end reproducibility. We showcase its value by benchmarking representative defenses across varied poisoning scenarios and tasks. By standardizing data, code, and metrics, SafeTuneBed is the first focused toolkit of its kind to accelerate rigorous and comparable research in safe LLM fine-tuning. Code is available at: https://github.com/criticalml-uw/SafeTuneBed

摘要: 随着大型语言模型（LLM）变得无处不在，参数高效的微调方法和安全第一的防御措施迅速增加。然而，方法的数量及其最近的增加导致了不同的评估-不同的数据集，指标和不一致的威胁设置-使得很难公平地比较各种方法的安全性，实用性和鲁棒性。为了解决这个问题，我们引入了SafeTuneBed，这是一个统一微调和防御评估的基准和工具包。SafeTuneBed（i）管理多个微调数据集的多样化存储库，涵盖情感分析，问答，多步推理和开放式指令任务，并允许生成有害变体分裂;（ii）实现最先进防御的集成，包括免疫阶段免疫，训练中保障措施和调优后修复;以及（iii）提供安全性（攻击成功率、拒绝一致性）和实用性的评估者。SafeTuneBed构建在Python优先、椭圆形驱动的脚本和插件之上，只需最少的额外代码即可指定任何微调机制、防御方法和指标套件，同时确保端到端的可重复性。我们通过对各种中毒场景和任务的代表性防御进行基准测试来展示其价值。通过标准化数据、代码和指标，SafeTuneBed是同类中第一个加速安全LLM微调方面严格且可比的研究的专注工具包。代码可访问：https://github.com/criticalml-uw/SafeTuneBed



## **9. SafeTy Reasoning Elicitation Alignment for Multi-Turn Dialogues**

多轮对话的安全推理启发对齐 cs.CL

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00668v1) [paper-pdf](http://arxiv.org/pdf/2506.00668v1)

**Authors**: Martin Kuo, Jianyi Zhang, Aolin Ding, Louis DiValentin, Amin Hass, Benjamin F Morris, Isaac Jacobson, Randolph Linderman, James Kiessling, Nicolas Ramos, Bhavna Gopal, Maziyar Baran Pouyan, Changwei Liu, Hai Li, Yiran Chen

**Abstract**: Malicious attackers can exploit large language models (LLMs) by engaging them in multi-turn dialogues to achieve harmful objectives, posing significant safety risks to society. To address this challenge, we propose a novel defense mechanism: SafeTy Reasoning Elicitation Alignment for Multi-Turn Dialogues (STREAM). STREAM defends LLMs against multi-turn attacks while preserving their functional capabilities. Our approach involves constructing a human-annotated dataset, the Safety Reasoning Multi-turn Dialogues dataset, which is used to fine-tune a plug-and-play safety reasoning moderator. This model is designed to identify malicious intent hidden within multi-turn conversations and alert the target LLM of potential risks. We evaluate STREAM across multiple LLMs against prevalent multi-turn attack strategies. Experimental results demonstrate that our method significantly outperforms existing defense techniques, reducing the Attack Success Rate (ASR) by 51.2%, all while maintaining comparable LLM capability.

摘要: 恶意攻击者可以通过让大型语言模型（LLM）参与多轮对话来利用它们来实现有害目标，从而对社会构成重大安全风险。为了应对这一挑战，我们提出了一种新颖的防御机制：SafeTy Reasoning启发式对齐多转弯对话（UTE）。MBE保护LLM免受多回合攻击，同时保留其功能能力。我们的方法涉及构建人类注释的数据集，即安全推理多轮对话数据集，用于微调即插即用安全推理主持人。该模型旨在识别隐藏在多轮对话中的恶意意图，并向目标LLM警告潜在风险。我们针对流行的多回合攻击策略评估多个LLM的MBE。实验结果表明，我们的方法显着优于现有的防御技术，将攻击成功率（ASB）降低了51.2%，同时保持了相当的LLM能力。



## **10. Stepwise Reasoning Error Disruption Attack of LLMs**

LLM的逐步推理错误中断攻击 cs.AI

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2412.11934v4) [paper-pdf](http://arxiv.org/pdf/2412.11934v4)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications. Our code is available at: https://github.com/Applied-Machine-Learning-Lab/SEED-Attack.

摘要: 大型语言模型（LLM）在复杂推理任务中取得了显着的进步，但其在推理过程中的安全性和稳健性仍然没有得到充分的探索。对LLM推理的现有攻击受到特定设置或缺乏不可感知性的限制，限制了其可行性和可概括性。为了解决这些挑战，我们提出了Stepwise rEasying错误破坏（SEED）攻击，它巧妙地将错误注入到先前的推理步骤中，以误导模型产生错误的后续推理和最终答案。与以前的方法不同，SEED与零镜头和少镜头设置兼容，保持自然推理流程，并确保在不修改指令的情况下隐蔽执行。对四个不同模型的四个数据集进行的广泛实验证明了SEED的有效性，揭示了LLM对推理过程中断的脆弱性。这些发现强调需要更加关注LLM推理的稳健性，以确保实际应用中的安全性。我们的代码可访问：https://github.com/Applied-Machine-Learning-Lab/SEED-Attack。



## **11. Goal-Aware Identification and Rectification of Misinformation in Multi-Agent Systems**

多代理系统中错误信息的目标感知识别和纠正 cs.CL

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00509v1) [paper-pdf](http://arxiv.org/pdf/2506.00509v1)

**Authors**: Zherui Li, Yan Mi, Zhenhong Zhou, Houcheng Jiang, Guibin Zhang, Kun Wang, Junfeng Fang

**Abstract**: Large Language Model-based Multi-Agent Systems (MASs) have demonstrated strong advantages in addressing complex real-world tasks. However, due to the introduction of additional attack surfaces, MASs are particularly vulnerable to misinformation injection. To facilitate a deeper understanding of misinformation propagation dynamics within these systems, we introduce MisinfoTask, a novel dataset featuring complex, realistic tasks designed to evaluate MAS robustness against such threats. Building upon this, we propose ARGUS, a two-stage, training-free defense framework leveraging goal-aware reasoning for precise misinformation rectification within information flows. Our experiments demonstrate that in challenging misinformation scenarios, ARGUS exhibits significant efficacy across various injection attacks, achieving an average reduction in misinformation toxicity of approximately 28.17% and improving task success rates under attack by approximately 10.33%. Our code and dataset is available at: https://github.com/zhrli324/ARGUS.

摘要: 基于大型语言模型的多代理系统（MAS）在解决复杂的现实世界任务方面表现出强大的优势。然而，由于引入了额外的攻击面，MAS特别容易受到错误信息注入的影响。为了促进对这些系统内错误信息传播动态的更深入了解，我们引入了MisinfoTask，这是一种新颖的数据集，具有复杂、现实的任务，旨在评估MAS针对此类威胁的稳健性。在此基础上，我们提出了ARUTE，这是一个两阶段、免训练的防御框架，利用目标感知推理来在信息流中进行精确的错误信息纠正。我们的实验表明，在具有挑战性的错误信息场景中，ARGucci在各种注入攻击中表现出显着的功效，实现了错误信息毒性平均降低约28.17%，并将攻击下的任务成功率提高约10.33%。我们的代码和数据集可访问：https://github.com/zhrli324/ARGUS。



## **12. WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermarks**

WET：用线性转换水印克服嵌入即服务中的漏洞解释 cs.CR

Accepted to ACL 2025 (Main Proceedings)

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2409.04459v2) [paper-pdf](http://arxiv.org/pdf/2409.04459v2)

**Authors**: Anudeex Shetty, Qiongkai Xu, Jey Han Lau

**Abstract**: Embeddings-as-a-Service (EaaS) is a service offered by large language model (LLM) developers to supply embeddings generated by LLMs. Previous research suggests that EaaS is prone to imitation attacks -- attacks that clone the underlying EaaS model by training another model on the queried embeddings. As a result, EaaS watermarks are introduced to protect the intellectual property of EaaS providers. In this paper, we first show that existing EaaS watermarks can be removed by paraphrasing when attackers clone the model. Subsequently, we propose a novel watermarking technique that involves linearly transforming the embeddings, and show that it is empirically and theoretically robust against paraphrasing.

摘要: 嵌入即服务（EASS）是由大型语言模型（LLM）开发人员提供的一项服务，用于提供LLM生成的嵌入。之前的研究表明，ESaaS容易受到模仿攻击--通过在查询的嵌入上训练另一个模型来克隆底层ESaaS模型的攻击。因此，引入了ESaaS水印来保护ESaaS提供商的知识产权。在本文中，我们首先表明，当攻击者克隆模型时，可以通过解释来删除现有的EASS水印。随后，我们提出了一种新颖的水印技术，该技术涉及线性变换嵌入，并表明它在经验和理论上都具有鲁棒性，对抗重述。



## **13. Spectral Insights into Data-Oblivious Critical Layers in Large Language Models**

对大型语言模型中数据不经意关键层的光谱洞察 cs.LG

Accepted by Findings of ACL2025

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00382v1) [paper-pdf](http://arxiv.org/pdf/2506.00382v1)

**Authors**: Xuyuan Liu, Lei Hsiung, Yaoqing Yang, Yujun Yan

**Abstract**: Understanding how feature representations evolve across layers in large language models (LLMs) is key to improving their interpretability and robustness. While recent studies have identified critical layers linked to specific functions or behaviors, these efforts typically rely on data-dependent analyses of fine-tuned models, limiting their use to post-hoc settings. In contrast, we introduce a data-oblivious approach to identify intrinsic critical layers in pre-fine-tuned LLMs by analyzing representation dynamics via Centered Kernel Alignment(CKA). We show that layers with significant shifts in representation space are also those most affected during fine-tuning--a pattern that holds consistently across tasks for a given model. Our spectral analysis further reveals that these shifts are driven by changes in the top principal components, which encode semantic transitions from rationales to conclusions. We further apply these findings to two practical scenarios: efficient domain adaptation, where fine-tuning critical layers leads to greater loss reduction compared to non-critical layers; and backdoor defense, where freezing them reduces attack success rates by up to 40%.

摘要: 了解特征表示如何在大型语言模型（LLM）中跨层演变是提高其可解释性和稳健性的关键。虽然最近的研究已经确定了与特定功能或行为相关的关键层，但这些工作通常依赖于对微调模型的数据依赖性分析，将其使用限制在事后环境中。相比之下，我们引入了一种数据不受关注的方法，通过通过中心核心对齐（CKA）分析表示动态来识别预微调的LLM中的内在关键层。我们表明，表示空间发生显着变化的层也是微调期间受影响最大的层--对于给定模型，这种模式在任务中始终保持一致。我们的谱分析进一步揭示了这些变化是由顶部主成分的变化驱动的，这些主成分编码了从基本原理到结论的语义转换。我们进一步将这些发现应用于两个实际场景：高效的域适应，与非关键层相比，微调关键层可以减少更大的损失;后门防御，冻结它们可以将攻击成功率降低高达40%。



## **14. Keeping an Eye on LLM Unlearning: The Hidden Risk and Remedy**

关注法学硕士遗忘：隐藏的风险和补救措施 cs.CR

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00359v1) [paper-pdf](http://arxiv.org/pdf/2506.00359v1)

**Authors**: Jie Ren, Zhenwei Dai, Xianfeng Tang, Yue Xing, Shenglai Zeng, Hui Liu, Jingying Zeng, Qiankun Peng, Samarth Varshney, Suhang Wang, Qi He, Charu C. Aggarwal, Hui Liu

**Abstract**: Although Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of tasks, growing concerns have emerged over the misuse of sensitive, copyrighted, or harmful data during training. To address these concerns, unlearning techniques have been developed to remove the influence of specific data without retraining from scratch. However, this paper reveals a critical vulnerability in fine-tuning-based unlearning: a malicious user can craft a manipulated forgetting request that stealthily degrades the model's utility for benign users. We demonstrate this risk through a red-teaming Stealthy Attack (SA), which is inspired by two key limitations of existing unlearning (the inability to constrain the scope of unlearning effect and the failure to distinguish benign tokens from unlearning signals). Prior work has shown that unlearned models tend to memorize forgetting data as unlearning signals, and respond with hallucinations or feigned ignorance when unlearning signals appear in the input. By subtly increasing the presence of common benign tokens in the forgetting data, SA enhances the connection between benign tokens and unlearning signals. As a result, when normal users include such tokens in their prompts, the model exhibits unlearning behaviors, leading to unintended utility degradation. To address this vulnerability, we propose Scope-aware Unlearning (SU), a lightweight enhancement that introduces a scope term into the unlearning objective, encouraging the model to localize the forgetting effect. Our method requires no additional data processing, integrates seamlessly with existing fine-tuning frameworks, and significantly improves robustness against SA. Extensive experiments validate the effectiveness of both SA and SU.

摘要: 尽管大型语言模型（LLM）在广泛的任务中表现出了令人印象深刻的能力，但人们越来越担心培训期间滥用敏感、受版权或有害数据。为了解决这些问题，人们开发了消除学习技术，以消除特定数据的影响，而无需从头开始重新培训。然而，本文揭示了基于微调的取消学习中的一个关键漏洞：恶意用户可以精心设计一个经过操纵的忘记请求，从而悄悄降低模型对良性用户的实用性。我们通过红组隐形攻击（SA）来证明这种风险，该攻击的灵感来自于现有取消学习的两个关键局限性（无法限制取消学习效果的范围以及无法区分良性令牌和取消学习信号）。之前的工作表明，未学习的模型倾向于将遗忘数据记忆为未学习信号，并在输入中出现未学习信号时以幻觉或假装无知做出反应。通过巧妙地增加遗忘数据中常见良性标记的存在，SA增强了良性标记和取消学习信号之间的联系。因此，当普通用户在其提示中包含此类令牌时，模型会表现出忘记学习行为，从而导致意外的效用下降。为了解决这个漏洞，我们提出了Scope-aware Unlearning（SU），这是一种轻量级增强，将范围术语引入到Unlearning目标中，鼓励模型本地化遗忘效应。我们的方法不需要额外的数据处理，与现有的微调框架无缝集成，并显着提高了针对SA的鲁棒性。大量实验验证了SA和SU的有效性。



## **15. Adversarial Threat Vectors and Risk Mitigation for Retrieval-Augmented Generation Systems**

检索增强生成系统的对抗性威胁向量和风险缓解 cs.CR

SPIE DCS: Proceedings Volume Assurance and Security for AI-enabled  Systems 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2506.00281v1) [paper-pdf](http://arxiv.org/pdf/2506.00281v1)

**Authors**: Chris M. Ward, Josh Harguess

**Abstract**: Retrieval-Augmented Generation (RAG) systems, which integrate Large Language Models (LLMs) with external knowledge sources, are vulnerable to a range of adversarial attack vectors. This paper examines the importance of RAG systems through recent industry adoption trends and identifies the prominent attack vectors for RAG: prompt injection, data poisoning, and adversarial query manipulation. We analyze these threats under risk management lens, and propose robust prioritized control list that includes risk-mitigating actions like input validation, adversarial training, and real-time monitoring.

摘要: 检索增强生成（RAG）系统将大型语言模型（LLM）与外部知识源集成，容易受到一系列对抗攻击载体的攻击。本文通过最近的行业采用趋势来探讨RAG系统的重要性，并确定了RAG的主要攻击载体：提示注入、数据中毒和对抗性查询操纵。我们在风险管理的视角下分析这些威胁，并提出强大的优先级控制列表，其中包括输入验证、对抗性培训和实时监控等风险缓解行动。



## **16. Differential privacy enables fair and accurate AI-based analysis of speech disorders while protecting patient data**

差异隐私能够公平准确地对言语障碍进行基于人工智能的分析，同时保护患者数据 cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2409.19078v3) [paper-pdf](http://arxiv.org/pdf/2409.19078v3)

**Authors**: Soroosh Tayebi Arasteh, Mahshad Lotfinia, Paula Andrea Perez-Toro, Tomas Arias-Vergara, Mahtab Ranji, Juan Rafael Orozco-Arroyave, Maria Schuster, Andreas Maier, Seung Hee Yang

**Abstract**: Speech pathology has impacts on communication abilities and quality of life. While deep learning-based models have shown potential in diagnosing these disorders, the use of sensitive data raises critical privacy concerns. Although differential privacy (DP) has been explored in the medical imaging domain, its application in pathological speech analysis remains largely unexplored despite the equally critical privacy concerns. To the best of our knowledge, this study is the first to investigate DP's impact on pathological speech data, focusing on the trade-offs between privacy, diagnostic accuracy, and fairness. Using a large, real-world dataset of 200 hours of recordings from 2,839 German-speaking participants, we observed a maximum accuracy reduction of 3.85% when training with DP with high privacy levels. To highlight real-world privacy risks, we demonstrated the vulnerability of non-private models to gradient inversion attacks, reconstructing identifiable speech samples and showcasing DP's effectiveness in mitigating these risks. To explore the potential generalizability across languages and disorders, we validated our approach on a dataset of Spanish-speaking Parkinson's disease patients, leveraging pretrained models from healthy English-speaking datasets, and demonstrated that careful pretraining on large-scale task-specific datasets can maintain favorable accuracy under DP constraints. A comprehensive fairness analysis revealed minimal gender bias at reasonable privacy levels but underscored the need for addressing age-related disparities. Our results establish that DP can balance privacy and utility in speech disorder detection, while highlighting unique challenges in privacy-fairness trade-offs for speech data. This provides a foundation for refining DP methodologies and improving fairness across diverse patient groups in real-world deployments.

摘要: 言语病理会影响沟通能力和生活质量。虽然基于深度学习的模型在诊断这些疾病方面表现出了潜力，但敏感数据的使用引发了严重的隐私问题。尽管差异隐私（DP）在医学成像领域得到了探索，但尽管隐私问题同样重要，但其在病理言语分析中的应用在很大程度上仍未被探索。据我们所知，这项研究是第一个调查DP对病理言语数据影响的研究，重点关注隐私、诊断准确性和公平性之间的权衡。使用包含2，839名德语参与者200小时录音的大型现实世界数据集，我们观察到使用隐私级别高的DP进行训练时，准确性最大下降了3.85%。为了强调现实世界的隐私风险，我们展示了非私有模型对梯度倒置攻击的脆弱性，重建了可识别的语音样本并展示了DP在减轻这些风险方面的有效性。为了探索跨语言和障碍的潜在概括性，我们在讲西班牙语的帕金森病患者数据集上验证了我们的方法，利用来自健康英语数据集的预训练模型，并证明对大规模特定任务的数据集进行仔细的预训练可以在DP约束下保持良好的准确性。全面的公平性分析显示，在合理的隐私水平上，性别偏见最小，但强调了解决与年龄相关的差异的必要性。我们的结果表明，DP可以平衡言语障碍检测中的隐私和实用性，同时强调语音数据隐私与公平权衡方面的独特挑战。这为完善DP方法并提高现实世界部署中不同患者群体的公平性提供了基础。



## **17. TRIDENT: Enhancing Large Language Model Safety with Tri-Dimensional Diversified Red-Teaming Data Synthesis**

TRIDENT：通过三维多元化红色团队数据合成增强大型语言模型安全性 cs.CL

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24672v1) [paper-pdf](http://arxiv.org/pdf/2505.24672v1)

**Authors**: Xiaorui Wu, Xiaofeng Mao, Fei Li, Xin Zhang, Xuanhong Li, Chong Teng, Donghong Ji, Zhuang Li

**Abstract**: Large Language Models (LLMs) excel in various natural language processing tasks but remain vulnerable to generating harmful content or being exploited for malicious purposes. Although safety alignment datasets have been introduced to mitigate such risks through supervised fine-tuning (SFT), these datasets often lack comprehensive risk coverage. Most existing datasets focus primarily on lexical diversity while neglecting other critical dimensions. To address this limitation, we propose a novel analysis framework to systematically measure the risk coverage of alignment datasets across three essential dimensions: Lexical Diversity, Malicious Intent, and Jailbreak Tactics. We further introduce TRIDENT, an automated pipeline that leverages persona-based, zero-shot LLM generation to produce diverse and comprehensive instructions spanning these dimensions. Each harmful instruction is paired with an ethically aligned response, resulting in two datasets: TRIDENT-Core, comprising 26,311 examples, and TRIDENT-Edge, with 18,773 examples. Fine-tuning Llama 3.1-8B on TRIDENT-Edge demonstrates substantial improvements, achieving an average 14.29% reduction in Harm Score, and a 20% decrease in Attack Success Rate compared to the best-performing baseline model fine-tuned on the WildBreak dataset.

摘要: 大型语言模型（LLM）在各种自然语言处理任务中表现出色，但仍然容易生成有害内容或被用于恶意目的。尽管引入了安全对齐数据集以通过监督微调（SFT）来减轻此类风险，但这些数据集通常缺乏全面的风险覆盖范围。大多数现有的数据集主要关注词汇多样性，而忽视了其他关键维度。为了解决这一局限性，我们提出了一种新颖的分析框架，来系统地衡量对齐数据集在三个基本维度上的风险覆盖范围：词汇多样性、恶意意图和越狱策略。我们进一步介绍TRIDENT，这是一个自动化的管道，它利用基于角色的零触发LLM生成来生成跨越这些维度的多样化和全面的指令。每个有害指令都与道德上一致的响应配对，产生两个数据集：TRIDENT-Core，包括26，311个示例，TRIDENT-Edge，包括18，773个示例。在TRIDENT-Edge上对Llama 3.1-8B进行微调，与在WildBreak数据集上进行微调的最佳基线模型相比，其伤害分数平均降低了14.29%，攻击成功率降低了20%。



## **18. Benchmarking Large Language Models for Cryptanalysis and Mismatched-Generalization**

为密码分析和不匹配概括的大型语言模型进行基准测试 cs.CL

Preprint

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24621v1) [paper-pdf](http://arxiv.org/pdf/2505.24621v1)

**Authors**: Utsav Maskey, Chencheng Zhu, Usman Naseem

**Abstract**: Recent advancements in Large Language Models (LLMs) have transformed natural language understanding and generation, leading to extensive benchmarking across diverse tasks. However, cryptanalysis a critical area for data security and encryption has not yet been thoroughly explored in LLM evaluations. To address this gap, we evaluate cryptanalytic potential of state of the art LLMs on encrypted texts generated using a range of cryptographic algorithms. We introduce a novel benchmark dataset comprising diverse plain texts spanning various domains, lengths, writing styles, and topics paired with their encrypted versions. Using zero-shot and few shot settings, we assess multiple LLMs for decryption accuracy and semantic comprehension across different encryption schemes. Our findings reveal key insights into the strengths and limitations of LLMs in side-channel communication while raising concerns about their susceptibility to jailbreaking attacks. This research highlights the dual-use nature of LLMs in security contexts and contributes to the ongoing discussion on AI safety and security.

摘要: 大型语言模型（LLM）的最新进展改变了自然语言的理解和生成，导致跨不同任务的广泛基准测试。然而，密码分析是数据安全和加密的关键领域，尚未在LLM评估中得到彻底探索。为了解决这一差距，我们评估了最先进的LLM对使用一系列加密算法生成的加密文本的加密分析潜力。我们引入了一个新颖的基准数据集，其中包括跨越不同领域、长度、写作风格和主题的多样化纯文本，并与其加密版本配对。使用零触发和少触发设置，我们评估多个LLM的解密准确性和不同加密方案的语义理解。我们的研究结果揭示了对LLM在侧渠道沟通中的优势和局限性的关键见解，同时引发了人们对它们容易受到越狱攻击的担忧。这项研究强调了LLM在安全环境中的双重用途性质，并有助于正在进行的关于人工智能安全性的讨论。



## **19. BaxBench: Can LLMs Generate Correct and Secure Backends?**

收件箱长凳：LLM能否生成正确且安全的后台？ cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2502.11844v3) [paper-pdf](http://arxiv.org/pdf/2502.11844v3)

**Authors**: Mark Vero, Niels Mündler, Victor Chibotaru, Veselin Raychev, Maximilian Baader, Nikola Jovanović, Jingxuan He, Martin Vechev

**Abstract**: Automatic program generation has long been a fundamental challenge in computer science. Recent benchmarks have shown that large language models (LLMs) can effectively generate code at the function level, make code edits, and solve algorithmic coding tasks. However, to achieve full automation, LLMs should be able to generate production-quality, self-contained application modules. To evaluate the capabilities of LLMs in solving this challenge, we introduce BaxBench, a novel evaluation benchmark consisting of 392 tasks for the generation of backend applications. We focus on backends for three critical reasons: (i) they are practically relevant, building the core components of most modern web and cloud software, (ii) they are difficult to get right, requiring multiple functions and files to achieve the desired functionality, and (iii) they are security-critical, as they are exposed to untrusted third-parties, making secure solutions that prevent deployment-time attacks an imperative. BaxBench validates the functionality of the generated applications with comprehensive test cases, and assesses their security exposure by executing end-to-end exploits. Our experiments reveal key limitations of current LLMs in both functionality and security: (i) even the best model, OpenAI o1, achieves a mere 62% on code correctness; (ii) on average, we could successfully execute security exploits on around half of the correct programs generated by each LLM; and (iii) in less popular backend frameworks, models further struggle to generate correct and secure applications. Progress on BaxBench signifies important steps towards autonomous and secure software development with LLMs.

摘要: 自动程序生成长期以来一直是计算机科学的一个根本挑战。最近的基准测试表明，大型语言模型（LLM）可以有效地生成功能级别的代码、进行代码编辑并解决算法编码任务。然而，为了实现完全自动化，LLM应该能够生成生产质量的独立应用程序模块。为了评估LLM解决这一挑战的能力，我们引入了DeliverBench，这是一个新颖的评估基准，由392个用于生成后台应用程序的任务组成。我们关注后端有三个关键原因：（i）它们实际上是相关的，构建了大多数现代Web和云软件的核心组件，（ii）它们很难正确，需要多个功能和文件来实现所需的功能，（iii）它们是安全关键的，因为它们暴露给不受信任的第三方，因此必须采取安全解决方案来防止部署时的攻击。CNORTBench使用全面的测试用例验证生成的应用程序的功能，并通过执行端到端漏洞来评估其安全风险。我们的实验揭示了当前LLM在功能和安全性方面的关键限制：（i）即使是最好的模型OpenAI o 1，代码正确性也只有62%;（ii）平均而言，我们可以在每个LLM生成的大约一半的正确程序上成功执行安全漏洞;（iii）在不太流行的后端框架中，模型进一步努力生成正确和安全的应用程序。在LLM Bench上取得的进展标志着LLM向自主和安全的软件开发迈出了重要的一步。



## **20. Stress-testing Machine Generated Text Detection: Shifting Language Models Writing Style to Fool Detectors**

压力测试机器生成文本检测：改变语言模型写作风格以愚弄检测器 cs.CL

Accepted at Findings of ACL 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24523v1) [paper-pdf](http://arxiv.org/pdf/2505.24523v1)

**Authors**: Andrea Pedrotti, Michele Papucci, Cristiano Ciaccio, Alessio Miaschi, Giovanni Puccetti, Felice Dell'Orletta, Andrea Esuli

**Abstract**: Recent advancements in Generative AI and Large Language Models (LLMs) have enabled the creation of highly realistic synthetic content, raising concerns about the potential for malicious use, such as misinformation and manipulation. Moreover, detecting Machine-Generated Text (MGT) remains challenging due to the lack of robust benchmarks that assess generalization to real-world scenarios. In this work, we present a pipeline to test the resilience of state-of-the-art MGT detectors (e.g., Mage, Radar, LLM-DetectAIve) to linguistically informed adversarial attacks. To challenge the detectors, we fine-tune language models using Direct Preference Optimization (DPO) to shift the MGT style toward human-written text (HWT). This exploits the detectors' reliance on stylistic clues, making new generations more challenging to detect. Additionally, we analyze the linguistic shifts induced by the alignment and which features are used by detectors to detect MGT texts. Our results show that detectors can be easily fooled with relatively few examples, resulting in a significant drop in detection performance. This highlights the importance of improving detection methods and making them robust to unseen in-domain texts.

摘要: 生成式人工智能和大型语言模型（LLM）的最新进展使得能够创建高度真实的合成内容，这引发了人们对恶意使用可能性的担忧，例如错误信息和操纵。此外，由于缺乏评估对现实世界场景的概括性的稳健基准，检测机器生成文本（MGT）仍然具有挑战性。在这项工作中，我们提出了一个管道来测试最先进的MGT检测器（例如，Mage、Radar、LLM-DetectAIve）到语言知情的对抗性攻击。为了挑战检测器，我们使用直接偏好优化（DPO）微调语言模型，将MGT风格转变为手写文本（HWT）。这利用了探测器对风格线索的依赖，使新一代的探测更具挑战性。此外，我们还分析了对齐引起的语言变化以及检测器使用哪些特征来检测MGT文本。我们的结果表明，检测器很容易被相对较少的例子所愚弄，导致检测性能显着下降。这凸显了改进检测方法并使其对不可见的领域内文本具有鲁棒性的重要性。



## **21. An Interpretable N-gram Perplexity Threat Model for Large Language Model Jailbreaks**

针对大型语言模型越狱的可解释N-gram困惑威胁模型 cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2410.16222v2) [paper-pdf](http://arxiv.org/pdf/2410.16222v2)

**Authors**: Valentyn Boreiko, Alexander Panfilov, Vaclav Voracek, Matthias Hein, Jonas Geiping

**Abstract**: A plethora of jailbreaking attacks have been proposed to obtain harmful responses from safety-tuned LLMs. These methods largely succeed in coercing the target output in their original settings, but their attacks vary substantially in fluency and computational effort. In this work, we propose a unified threat model for the principled comparison of these methods. Our threat model checks if a given jailbreak is likely to occur in the distribution of text. For this, we build an N-gram language model on 1T tokens, which, unlike model-based perplexity, allows for an LLM-agnostic, nonparametric, and inherently interpretable evaluation. We adapt popular attacks to this threat model, and, for the first time, benchmark these attacks on equal footing with it. After an extensive comparison, we find attack success rates against safety-tuned modern models to be lower than previously presented and that attacks based on discrete optimization significantly outperform recent LLM-based attacks. Being inherently interpretable, our threat model allows for a comprehensive analysis and comparison of jailbreak attacks. We find that effective attacks exploit and abuse infrequent bigrams, either selecting the ones absent from real-world text or rare ones, e.g., specific to Reddit or code datasets.

摘要: 人们提出了大量越狱攻击，以从经过安全调整的LLM获得有害响应。这些方法在很大程度上成功地将目标输出强制到其原始设置中，但它们的攻击在流畅性和计算工作量方面存在很大差异。在这项工作中，我们提出了一个统一的威胁模型，用于对这些方法进行原则性比较。我们的威胁模型检查特定越狱是否可能发生在文本分发中。为此，我们在1 T令牌上构建了一个N-gram语言模型，与基于模型的困惑不同，它允许LLM不可知、非参数且本质上可解释的评估。我们将流行的攻击适应这种威胁模型，并首次将这些攻击与其同等地位进行基准测试。经过广泛比较，我们发现针对安全优化的现代模型的攻击成功率低于之前提出的，并且基于离散优化的攻击显着优于最近的基于LLM的攻击。我们的威胁模型本质上是可解释的，允许对越狱攻击进行全面分析和比较。我们发现，有效的攻击会利用和滥用不常见的二元语法，要么选择现实世界文本中缺失的二元语法，要么选择罕见的二元语法，例如，特定于Reddit或代码数据集。



## **22. SEAR: A Multimodal Dataset for Analyzing AR-LLM-Driven Social Engineering Behaviors**

SEAR：用于分析AR-LLM驱动的社会工程行为的多模式数据集 cs.AI

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24458v1) [paper-pdf](http://arxiv.org/pdf/2505.24458v1)

**Authors**: Tianlong Yu, Chenghang Ye, Zheyu Yang, Ziyi Zhou, Cui Tang, Zui Tao, Jun Zhang, Kailong Wang, Liting Zhou, Yang Yang, Ting Bi

**Abstract**: The SEAR Dataset is a novel multimodal resource designed to study the emerging threat of social engineering (SE) attacks orchestrated through augmented reality (AR) and multimodal large language models (LLMs). This dataset captures 180 annotated conversations across 60 participants in simulated adversarial scenarios, including meetings, classes and networking events. It comprises synchronized AR-captured visual/audio cues (e.g., facial expressions, vocal tones), environmental context, and curated social media profiles, alongside subjective metrics such as trust ratings and susceptibility assessments. Key findings reveal SEAR's alarming efficacy in eliciting compliance (e.g., 93.3% phishing link clicks, 85% call acceptance) and hijacking trust (76.7% post-interaction trust surge). The dataset supports research in detecting AR-driven SE attacks, designing defensive frameworks, and understanding multimodal adversarial manipulation. Rigorous ethical safeguards, including anonymization and IRB compliance, ensure responsible use. The SEAR dataset is available at https://github.com/INSLabCN/SEAR-Dataset.

摘要: SEAR数据集是一种新型多模式资源，旨在研究通过增强现实（AR）和多模式大型语言模型（LLM）精心策划的社会工程（SE）攻击的新兴威胁。该数据集捕获了模拟对抗场景（包括会议、课程和网络活动）中60名参与者的180个带注释的对话。它包括同步的AR捕获的视觉/音频线索（例如，面部表情、语气）、环境背景和精心策划的社交媒体个人资料，以及信任评级和易感性评估等主观指标。关键发现揭示了SEAR在引发合规方面的惊人功效（例如，93.3%的网络钓鱼链接点击，85%的电话接受）和劫持信任（互动后信任激增76.7%）。该数据集支持检测AR驱动的SE攻击、设计防御框架和理解多模式对抗操纵的研究。严格的道德保障措施，包括匿名化和机构审查委员会合规性，确保负责任的使用。SEAR数据集可在https://github.com/INSLabCN/SEAR-Dataset上获取。



## **23. Learning Safety Constraints for Large Language Models**

大型语言模型的学习安全约束 cs.LG

ICML 2025 (Spotlight)

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24445v1) [paper-pdf](http://arxiv.org/pdf/2505.24445v1)

**Authors**: Xin Chen, Yarden As, Andreas Krause

**Abstract**: Large language models (LLMs) have emerged as powerful tools but pose significant safety risks through harmful outputs and vulnerability to adversarial attacks. We propose SaP, short for Safety Polytope, a geometric approach to LLM safety that learns and enforces multiple safety constraints directly in the model's representation space. We develop a framework that identifies safe and unsafe regions via the polytope's facets, enabling both detection and correction of unsafe outputs through geometric steering. Unlike existing approaches that modify model weights, SaP operates post-hoc in the representation space, preserving model capabilities while enforcing safety constraints. Experiments across multiple LLMs demonstrate that our method can effectively detect unethical inputs, reduce adversarial attack success rates while maintaining performance on standard tasks, thus highlighting the importance of having an explicit geometric model for safety. Analysis of the learned polytope facets reveals emergence of specialization in detecting different semantic notions of safety, providing interpretable insights into how safety is captured in LLMs' representation space.

摘要: 大型语言模型（LLM）已成为强大的工具，但由于有害输出和易受对抗攻击而构成重大安全风险。我们提出SaP（Safety Polytope的缩写），这是一种LLM安全性的几何方法，可以直接在模型的表示空间中学习和强制执行多个安全约束。我们开发了一个框架，通过多面体识别安全和不安全区域，从而通过几何转向检测和纠正不安全输出。与修改模型权重的现有方法不同，SaP在表示空间中事后操作，在强制执行安全约束的同时保留模型能力。跨多个LLM的实验表明，我们的方法可以有效地检测不道德的输入，降低对抗攻击成功率，同时保持标准任务的性能，从而凸显了拥有显式几何模型的重要性安全性。对习得的多格面的分析揭示了检测不同安全性语义概念的专业化的出现，从而为如何在LLM的表示空间中捕获安全性提供了可解释的见解。



## **24. Breaking the Gold Standard: Extracting Forgotten Data under Exact Unlearning in Large Language Models**

打破黄金标准：在大型语言模型中提取精确非学习下的遗忘数据 cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24379v1) [paper-pdf](http://arxiv.org/pdf/2505.24379v1)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large language models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard, believed to be robust against privacy-related attacks. In this paper, we challenge this assumption by introducing a novel data extraction attack that compromises even exact unlearning. Our method leverages both the pre- and post-unlearning models: by guiding the post-unlearning model using signals from the pre-unlearning model, we uncover patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints.

摘要: 大型语言模型通常在从网络收集的数据集上训练，这些数据集可能无意中包含有害或敏感的个人信息。为了解决日益增长的隐私问题，人们提出了取消学习方法来消除训练模型中特定数据的影响。其中，精确取消学习--在没有目标数据的情况下从头开始重新训练模型--被广泛认为是黄金标准，被认为对隐私相关攻击具有强大的鲁棒性。在本文中，我们通过引入一种新颖的数据提取攻击来挑战这一假设，该攻击甚至会损害精确的取消学习。我们的方法利用了取消学习前和取消学习后的模型：通过使用来自取消学习前模型的信号引导取消学习后模型，我们发现了反映已删除数据分布的模式。将模型指导与令牌过滤策略相结合，我们的攻击显着提高了MUSE、TOFU和WMDP等常见基准测试中的提取成功率，在某些情况下性能翻倍。此外，我们还展示了我们对模拟医疗诊断数据集的攻击的有效性，以强调与精确忘记相关的现实世界隐私风险。鉴于我们的研究结果表明，取消学习可能会以一种矛盾的方式增加隐私泄露的风险，我们主张对取消学习方法进行评估，以考虑更广泛的威胁模型，这些模型不仅考虑取消学习后的模型，还考虑到对之前检查点的对抗访问。



## **25. Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction**

重写越狱：发现可学习和可转移的隐性有害指令 cs.CL

22 pages, 10 figures, accepted to ACL 2025 findings

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2502.11084v2) [paper-pdf](http://arxiv.org/pdf/2502.11084v2)

**Authors**: Yuting Huang, Chengyuan Liu, Yifeng Feng, Yiquan Wu, Chao Wu, Fei Wu, Kun Kuang

**Abstract**: As Large Language Models (LLMs) are widely applied in various domains, the safety of LLMs is increasingly attracting attention to avoid their powerful capabilities being misused. Existing jailbreak methods create a forced instruction-following scenario, or search adversarial prompts with prefix or suffix tokens to achieve a specific representation manually or automatically. However, they suffer from low efficiency and explicit jailbreak patterns, far from the real deployment of mass attacks to LLMs. In this paper, we point out that simply rewriting the original instruction can achieve a jailbreak, and we find that this rewriting approach is learnable and transferable. We propose the Rewrite to Jailbreak (R2J) approach, a transferable black-box jailbreak method to attack LLMs by iteratively exploring the weakness of the LLMs and automatically improving the attacking strategy. The jailbreak is more efficient and hard to identify since no additional features are introduced. Extensive experiments and analysis demonstrate the effectiveness of R2J, and we find that the jailbreak is also transferable to multiple datasets and various types of models with only a few queries. We hope our work motivates further investigation of LLM safety. The code can be found at https://github.com/ythuang02/R2J/.

摘要: 随着大型语言模型（LLM）在各个领域的广泛应用，LLM的安全性越来越受到关注，以避免其强大的功能被滥用。现有的越狱方法创建强制描述跟随场景，或搜索具有前置或后缀标记的对抗提示，以手动或自动实现特定的表示。然而，它们的效率低和越狱模式明显，远未真正对LLM进行大规模攻击。在本文中，我们指出，简单地重写原始指令就可以实现越狱，并且我们发现这种重写方法是可学习和可移植的。我们提出了重写越狱（R2 J）方法，这是一种可转移的黑匣子越狱方法，通过迭代探索LLM的弱点并自动改进攻击策略来攻击LLM。由于没有引入额外的功能，越狱更有效且难以识别。大量的实验和分析证明了R2J的有效性，我们发现越狱也可以转移到多个数据集和各种类型的模型，只需几个查询。我们希望我们的工作能够推动对LLM安全性的进一步调查。该代码可在https://github.com/ythuang02/R2J/上找到。



## **26. A Reward-driven Automated Webshell Malicious-code Generator for Red-teaming**

用于红色团队的奖励驱动自动Webshell恶意代码生成器 cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24252v1) [paper-pdf](http://arxiv.org/pdf/2505.24252v1)

**Authors**: Yizhong Ding

**Abstract**: Frequent cyber-attacks have elevated WebShell exploitation and defense to a critical research focus within network security. However, there remains a significant shortage of publicly available, well-categorized malicious-code datasets organized by obfuscation method. Existing malicious-code generation methods, which primarily rely on prompt engineering, often suffer from limited diversity and high redundancy in the payloads they produce. To address these limitations, we propose \textbf{RAWG}, a \textbf{R}eward-driven \textbf{A}utomated \textbf{W}ebshell Malicious-code \textbf{G}enerator designed for red-teaming applications. Our approach begins by categorizing webshell samples from common datasets into seven distinct types of obfuscation. We then employ a large language model (LLM) to extract and normalize key tokens from each sample, creating a standardized, high-quality corpus. Using this curated dataset, we perform supervised fine-tuning (SFT) on an open-source large model to enable the generation of diverse, highly obfuscated webshell malicious payloads. To further enhance generation quality, we apply Proximal Policy Optimization (PPO), treating malicious-code samples as "chosen" data and benign code as "rejected" data during reinforcement learning. Extensive experiments demonstrate that RAWG significantly outperforms current state-of-the-art methods in both payload diversity and escape effectiveness.

摘要: 频繁的网络攻击已将WebShell的利用和防御提升为网络安全领域的关键研究焦点。然而，通过混淆方法组织的公开可用、分类良好的恶意代码数据集仍然严重短缺。现有的恶意代码生成方法主要依赖于即时工程，其产生的有效负载的多样性和高冗余性往往受到影响。为了解决这些限制，我们提出了\textBF{RAWG}，这是一个\textBF{R}奖励驱动的\textBF{A}utomated \textBF{W}ebshell恶意代码\textBF{G}生成器，专为红色团队应用程序设计。我们的方法首先将来自常见数据集的webshell样本分类为七种不同类型的混淆。然后，我们采用大型语言模型（LLM）从每个样本中提取和规范化关键标记，创建标准化、高质量的文集。使用这个精心策划的数据集，我们对开源大型模型执行监督微调（SFT），以生成多样化、高度模糊的webshell恶意负载。为了进一步提高生成质量，我们应用了近端策略优化（PPO），在强化学习期间将恶意代码样本视为“选择”数据，将良性代码视为“拒绝”数据。大量的实验表明，RAWG显着优于目前国家的最先进的方法，在有效载荷的多样性和逃逸的有效性。



## **27. Safety Alignment Can Be Not Superficial With Explicit Safety Signals**

明确的安全信号，安全调整不能肤浅 cs.CR

ICML 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.17072v2) [paper-pdf](http://arxiv.org/pdf/2505.17072v2)

**Authors**: Jianwei Li, Jung-Eun Kim

**Abstract**: Recent studies on the safety alignment of large language models (LLMs) have revealed that existing approaches often operate superficially, leaving models vulnerable to various adversarial attacks. Despite their significance, these studies generally fail to offer actionable solutions beyond data augmentation for achieving more robust safety mechanisms. This paper identifies a fundamental cause of this superficiality: existing alignment approaches often presume that models can implicitly learn a safety-related reasoning task during the alignment process, enabling them to refuse harmful requests. However, the learned safety signals are often diluted by other competing objectives, leading models to struggle with drawing a firm safety-conscious decision boundary when confronted with adversarial attacks. Based on this observation, by explicitly introducing a safety-related binary classification task and integrating its signals with our attention and decoding strategies, we eliminate this ambiguity and allow models to respond more responsibly to malicious queries. We emphasize that, with less than 0.2x overhead cost, our approach enables LLMs to assess the safety of both the query and the previously generated tokens at each necessary generating step. Extensive experiments demonstrate that our method significantly improves the resilience of LLMs against various adversarial attacks, offering a promising pathway toward more robust generative AI systems.

摘要: 最近关于大型语言模型（LLM）安全对齐的研究表明，现有方法通常是肤浅的，使模型容易受到各种对抗攻击。尽管这些研究意义重大，但通常无法提供数据增强之外的可操作解决方案来实现更强大的安全机制。本文指出了这种肤浅的根本原因：现有的对齐方法通常假设模型可以在对齐过程中隐式学习与安全相关的推理任务，使它们能够拒绝有害的请求。然而，学习到的安全信号通常会被其他竞争目标稀释，导致模型在面临对抗攻击时难以划定坚定的安全意识决策边界。基于这一观察，通过明确引入与安全相关的二进制分类任务，并将其信号与我们的注意力和解码策略集成，我们消除了这种模糊性，并允许模型更负责任地响应恶意查询。我们强调，由于管理成本低于0.2倍，我们的方法使LLM能够在每个必要的生成步骤中评估查询和之前生成的令牌的安全性。大量实验表明，我们的方法显着提高了LLM对各种对抗攻击的弹性，为实现更强大的生成性人工智能系统提供了一条有希望的途径。



## **28. Benchmarking Foundation Models for Zero-Shot Biometric Tasks**

零镜头生物识别任务的基准基础模型 cs.CV

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24214v1) [paper-pdf](http://arxiv.org/pdf/2505.24214v1)

**Authors**: Redwan Sony, Parisa Farmanifard, Hamzeh Alzwairy, Nitish Shukla, Arun Ross

**Abstract**: The advent of foundation models, particularly Vision-Language Models (VLMs) and Multi-modal Large Language Models (MLLMs), has redefined the frontiers of artificial intelligence, enabling remarkable generalization across diverse tasks with minimal or no supervision. Yet, their potential in biometric recognition and analysis remains relatively underexplored. In this work, we introduce a comprehensive benchmark that evaluates the zero-shot and few-shot performance of state-of-the-art publicly available VLMs and MLLMs across six biometric tasks spanning the face and iris modalities: face verification, soft biometric attribute prediction (gender and race), iris recognition, presentation attack detection (PAD), and face manipulation detection (morphs and deepfakes). A total of 41 VLMs were used in this evaluation. Experiments show that embeddings from these foundation models can be used for diverse biometric tasks with varying degrees of success. For example, in the case of face verification, a True Match Rate (TMR) of 96.77 percent was obtained at a False Match Rate (FMR) of 1 percent on the Labeled Face in the Wild (LFW) dataset, without any fine-tuning. In the case of iris recognition, the TMR at 1 percent FMR on the IITD-R-Full dataset was 97.55 percent without any fine-tuning. Further, we show that applying a simple classifier head to these embeddings can help perform DeepFake detection for faces, Presentation Attack Detection (PAD) for irides, and extract soft biometric attributes like gender and ethnicity from faces with reasonably high accuracy. This work reiterates the potential of pretrained models in achieving the long-term vision of Artificial General Intelligence.

摘要: 基础模型的出现，特别是视觉语言模型（VLMS）和多模式大型语言模型（MLLM），重新定义了人工智能的前沿，能够在最少或没有监督的情况下实现对不同任务的显着概括。然而，它们在生物识别和分析方面的潜力仍然相对未充分开发。在这项工作中，我们引入了一个全面的基准，该基准评估了最先进的公开VLM和MLLM在跨越面部和虹膜模式的六项生物识别任务中的零镜头和少镜头性能：面部验证、软生物识别属性预测（性别和种族）、虹膜识别、呈现攻击检测（PAD）和面部操纵检测（变形和深度伪造）。本次评估中总共使用了41个VLM。实验表明，这些基础模型的嵌入可以用于各种生物识别任务，并取得不同程度的成功。例如，在面部验证的情况下，在野外标签面部（LFW）数据集中，在1%的假匹配率（FMR）下获得了96.77%的真匹配率（TLR），无需进行任何微调。就虹膜识别而言，在没有任何微调的情况下，IITD-R-Full数据集上1% FMR时的TLR为97.55%。此外，我们表明，将简单的分类器头应用于这些嵌入可以帮助执行人脸的DeepFake检测、虹膜的呈现攻击检测（PAD），并以相当高的准确性从人脸中提取性别和种族等软生物识别属性。这项工作重申了预训练模型在实现人工通用智能长期愿景方面的潜力。



## **29. On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains**

检索增强生成在知识密集型应用领域中的脆弱性研究 cs.CR

Accepted by ICML 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2409.17275v2) [paper-pdf](http://arxiv.org/pdf/2409.17275v2)

**Authors**: Xun Xian, Ganghua Wang, Xuan Bi, Jayanth Srinivasa, Ashish Kundu, Charles Fleming, Mingyi Hong, Jie Ding

**Abstract**: Retrieval-Augmented Generation (RAG) has been empirically shown to enhance the performance of large language models (LLMs) in knowledge-intensive domains such as healthcare, finance, and legal contexts. Given a query, RAG retrieves relevant documents from a corpus and integrates them into the LLMs' generation process. In this study, we investigate the adversarial robustness of RAG, focusing specifically on examining the retrieval system. First, across 225 different setup combinations of corpus, retriever, query, and targeted information, we show that retrieval systems are vulnerable to universal poisoning attacks in medical Q\&A. In such attacks, adversaries generate poisoned documents containing a broad spectrum of targeted information, such as personally identifiable information. When these poisoned documents are inserted into a corpus, they can be accurately retrieved by any users, as long as attacker-specified queries are used. To understand this vulnerability, we discovered that the deviation from the query's embedding to that of the poisoned document tends to follow a pattern in which the high similarity between the poisoned document and the query is retained, thereby enabling precise retrieval. Based on these findings, we develop a new detection-based defense to ensure the safe use of RAG. Through extensive experiments spanning various Q\&A domains, we observed that our proposed method consistently achieves excellent detection rates in nearly all cases.

摘要: 经验证明，检索增强生成（RAG）可以增强大型语言模型（LLM）在医疗保健、金融和法律上下文等知识密集型领域的性能。给定一个查询，RAG从数据库中检索相关文档，并将它们集成到LLM的生成过程中。在这项研究中，我们研究了RAG的对抗鲁棒性，特别关注检查检索系统。首先，在225种不同的数据库、检索器、查询和目标信息的设置组合中，我们表明检索系统容易受到医学问答中的普遍中毒攻击。在此类攻击中，对手会生成包含广泛目标信息（例如个人可识别信息）的有毒文档。当这些有毒文档被插入到数据库中时，只要使用攻击者指定的查询，任何用户都可以准确地检索它们。为了了解这个漏洞，我们发现查询嵌入与中毒文档嵌入的偏差往往遵循这样一种模式，即中毒文档和查询之间的高度相似性被保留，从而实现精确的检索。基于这些发现，我们开发了一种新的基于检测的防御措施，以确保RAG的安全使用。通过跨越各个问答领域的广泛实验，我们观察到我们提出的方法在几乎所有情况下都能始终实现出色的检测率。



## **30. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

Agent Security Bench（ASB）：基于LLM的Agent中的形式化和基准化攻击和防御 cs.CR

Accepted by ICLR 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2410.02644v4) [paper-pdf](http://arxiv.org/pdf/2410.02644v4)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.

摘要: 尽管基于LLM的代理在大型语言模型（LLM）的支持下可以使用外部工具和内存机制来解决复杂的现实世界任务，但它们也可能引入关键的安全漏洞。然而，现有文献并未全面评估针对基于LLM的代理的攻击和防御。为了解决这个问题，我们引入了代理安全工作台（ASB），这是一个全面的框架，旨在形式化、基准化和评估基于LLM的代理的攻击和防御，包括10种场景（例如，电子商务、自动驾驶、金融）、10个针对场景的代理、400多种工具、27种不同类型的攻击/防御方法和7个评估指标。基于ASB，我们对10种提示注入攻击、一种记忆中毒攻击、一种新颖的思想计划后门攻击、4种混合攻击以及13个LLM主干上的11种相应防御进行了基准测试。我们的基准测试结果揭示了代理操作不同阶段的关键漏洞，包括系统提示、用户提示处理、工具使用和内存检索，平均攻击成功率最高，为84.30%，但当前防御中表现出的有效性有限，揭示了社区在代理安全方面需要做的重要工作。我们还引入了一个新的指标来评估代理平衡实用性和安全性的能力。我们的代码可在https://github.com/agiresearch/ASB上找到。



## **31. Latent-space adversarial training with post-aware calibration for defending large language models against jailbreak attacks**

具有事后感知校准的潜在空间对抗训练，用于保护大型语言模型免受越狱攻击 cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2501.10639v3) [paper-pdf](http://arxiv.org/pdf/2501.10639v3)

**Authors**: Xin Yi, Yue Li, Dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Ensuring safety alignment is a critical requirement for large language models (LLMs), particularly given increasing deployment in real-world applications. Despite considerable advancements, LLMs remain susceptible to jailbreak attacks, which exploit system vulnerabilities to circumvent safety measures and elicit harmful or inappropriate outputs. Furthermore, while adversarial training-based defense methods have shown promise, a prevalent issue is the unintended over-defense behavior, wherein models excessively reject benign queries, significantly undermining their practical utility. To address these limitations, we introduce LATPC, a Latent-space Adversarial Training with Post-aware Calibration framework. LATPC dynamically identifies safety-critical latent dimensions by contrasting harmful and benign inputs, enabling the adaptive construction of targeted refusal feature removal attacks. This mechanism allows adversarial training to concentrate on real-world jailbreak tactics that disguise harmful queries as benign ones. During inference, LATPC employs an efficient embedding-level calibration mechanism to minimize over-defense behaviors with negligible computational overhead. Experimental results across five types of disguise-based jailbreak attacks demonstrate that LATPC achieves a superior balance between safety and utility compared to existing defense frameworks. Further analysis demonstrates the effectiveness of leveraging safety-critical dimensions in developing robust defense methods against jailbreak attacks.

摘要: 确保安全一致是大型语言模型（LLM）的关键要求，特别是考虑到现实世界应用程序中的部署不断增加。尽管LLM取得了相当大的进步，但仍然容易受到越狱攻击，这些攻击利用系统漏洞来规避安全措施并引发有害或不当的输出。此外，虽然基于对抗训练的防御方法已经显示出希望，但一个普遍的问题是无意的过度防御行为，其中模型过度拒绝良性查询，从而显着削弱了其实际实用性。为了解决这些限制，我们引入了LAPC，这是一种具有事后感知校准框架的潜在空间对抗训练。LAPC通过对比有害和良性输入来动态识别对安全至关重要的潜在维度，从而能够自适应地构建有针对性的拒绝功能删除攻击。这种机制允许对抗训练集中在现实世界的越狱策略上，这些策略将有害查询伪装成良性查询。在推理过程中，LAPC采用高效的嵌入级校准机制，以可忽略的计算负担最小化过度防御行为。五种基于伪装的越狱攻击的实验结果表明，与现有防御框架相比，LAPC在安全性和实用性之间实现了更好的平衡。进一步的分析证明了利用安全关键维度来开发针对越狱攻击的强大防御方法的有效性。



## **32. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

X-Transfer攻击：CLIP上的超级可转移对抗攻击 cs.CV

ICML 2025

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.05528v3) [paper-pdf](http://arxiv.org/pdf/2505.05528v3)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.

摘要: 随着对比图像预训练（CLIP）模型越来越多地被用于各种下游任务并集成到大型视觉语言模型（VLM）中，它们对对抗性扰动的敏感性已成为一个关键问题。在这项工作中，我们介绍了\textbf{X-Transfer}，一种新的攻击方法，暴露了CLIP中的一个普遍的对抗性漏洞。X-Transfer生成一个通用对抗扰动（Universal Adversarial Perturbation，UAP），能够欺骗不同样本、任务和域中的各种CLIP编码器和下游VLM。我们将此属性称为\textbf{super transferability}--一个同时实现跨数据、跨域、跨模型和跨任务对抗性可转移性的单一扰动。这是通过\textBF{代理缩放}来实现的，这是我们方法的一个关键创新。与依赖于固定代理模型（扩展计算密集型）的现有方法不同，X-Transfer采用高效的代理扩展策略，可以从大搜索空间中动态选择合适代理的一小子集。广泛的评估表明，X-Transfer的性能显着优于之前最先进的UAP方法，为跨CLIP模型的对抗性可移植性建立了新的基准。该代码可在我们的\href{https：//github.com/HanxunH/XTransferBench}{GitHub存储库}中公开获取。



## **33. LLM Agents Should Employ Security Principles**

LLM代理应采用安全原则 cs.CR

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.24019v1) [paper-pdf](http://arxiv.org/pdf/2505.24019v1)

**Authors**: Kaiyuan Zhang, Zian Su, Pin-Yu Chen, Elisa Bertino, Xiangyu Zhang, Ninghui Li

**Abstract**: Large Language Model (LLM) agents show considerable promise for automating complex tasks using contextual reasoning; however, interactions involving multiple agents and the system's susceptibility to prompt injection and other forms of context manipulation introduce new vulnerabilities related to privacy leakage and system exploitation. This position paper argues that the well-established design principles in information security, which are commonly referred to as security principles, should be employed when deploying LLM agents at scale. Design principles such as defense-in-depth, least privilege, complete mediation, and psychological acceptability have helped guide the design of mechanisms for securing information systems over the last five decades, and we argue that their explicit and conscientious adoption will help secure agentic systems. To illustrate this approach, we introduce AgentSandbox, a conceptual framework embedding these security principles to provide safeguards throughout an agent's life-cycle. We evaluate with state-of-the-art LLMs along three dimensions: benign utility, attack utility, and attack success rate. AgentSandbox maintains high utility for its intended functions under both benign and adversarial evaluations while substantially mitigating privacy risks. By embedding secure design principles as foundational elements within emerging LLM agent protocols, we aim to promote trustworthy agent ecosystems aligned with user privacy expectations and evolving regulatory requirements.

摘要: 大型语言模型（LLM）代理使用上下文推理自动化复杂的任务显示出相当大的希望;然而，涉及多个代理的交互和系统对提示注入和其他形式的上下文操作的敏感性引入了与隐私泄露和系统利用相关的新漏洞。这份立场文件认为，在大规模部署LLM代理时，应采用信息安全中公认的设计原则，通常称为安全原则。设计原则，如纵深防御，最小的特权，完全调解，心理上的可接受性，帮助指导设计的机制，确保信息系统在过去的五十年，我们认为，他们明确和认真的采用将有助于安全代理系统。为了说明这种方法，我们引入AgentSandbox，一个概念框架嵌入这些安全原则，在整个代理的生命周期提供保障。我们沿着三个维度评估最先进的LLM：良性效用，攻击效用和攻击成功率。AgentSandbox在良性和对抗性评估下保持其预期功能的高实用性，同时大大降低隐私风险。通过在新兴的LLM代理协议中嵌入安全设计原则作为基本元素，我们的目标是促进与用户隐私期望和不断变化的监管要求相一致的值得信赖的代理生态系统。



## **34. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

SVIP：迈向开源大型语言模型的可验证推理 cs.LG

22 pages

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.22307v2) [paper-pdf](http://arxiv.org/pdf/2410.22307v2)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: The ever-increasing size of open-source Large Language Models (LLMs) renders local deployment impractical for individual users. Decentralized computing has emerged as a cost-effective solution, allowing individuals and small companies to perform LLM inference for users using surplus computational power. However, a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby benefiting from cost savings. We introduce SVIP, a secret-based verifiable LLM inference protocol. Unlike existing solutions based on cryptographic or game-theoretic techniques, our method is computationally effective and does not rest on strong assumptions. Our protocol requires the computing provider to return both the generated text and processed hidden representations from LLMs. We then train a proxy task on these representations, effectively transforming them into a unique model identifier. With our protocol, users can reliably verify whether the computing provider is acting honestly. A carefully integrated secret mechanism further strengthens its security. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per prompt query for verification.

摘要: 开源大型语言模型（LLM）的规模不断扩大，使得本地部署对于个人用户来说变得不切实际。去中心化计算已成为一种具有成本效益的解决方案，允许个人和小公司使用剩余计算能力为用户执行LLM推理。然而，计算提供商可能会在未经用户同意的情况下偷偷地用更小、功能较差的模型替换请求的LLM，从而受益于成本节约。我们引入了SVIP，这是一种基于秘密的可验证LLM推理协议。与现有的解决方案的基础上加密或博弈论技术，我们的方法是计算有效的，不依赖于强假设。我们的协议要求计算提供者从LLM返回生成的文本和处理后的隐藏表示。然后，我们在这些表示上训练代理任务，有效地将它们转换为唯一的模型标识符。使用我们的协议，用户可以可靠地验证计算提供商是否诚实行事。精心集成的秘密机制进一步加强了其安全性。我们在多种强大且自适应的对抗场景下彻底分析了我们的协议。我们广泛的实验表明，SVIP准确、可概括、计算效率高，并且可以抵抗各种攻击。值得注意的是，SVIP的假阴性率低于5%，假阳性率低于3%，同时每次提示查询需要不到0.01秒的时间进行验证。



## **35. MCP Safety Training: Learning to Refuse Falsely Benign MCP Exploits using Improved Preference Alignment**

LCP安全培训：学会使用改进的偏好对齐来拒绝虚假良性的LCP利用 cs.LG

27 pages, 19 figures, 4 tables

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23634v1) [paper-pdf](http://arxiv.org/pdf/2505.23634v1)

**Authors**: John Halloran

**Abstract**: The model context protocol (MCP) has been widely adapted as an open standard enabling the seamless integration of generative AI agents. However, recent work has shown the MCP is susceptible to retrieval-based "falsely benign" attacks (FBAs), allowing malicious system access and credential theft, but requiring that users download compromised files directly to their systems. Herein, we show that the threat model of MCP-based attacks is significantly broader than previously thought, i.e., attackers need only post malicious content online to deceive MCP agents into carrying out their attacks on unsuspecting victims' systems.   To improve alignment guardrails against such attacks, we introduce a new MCP dataset of FBAs and (truly) benign samples to explore the effectiveness of direct preference optimization (DPO) for the refusal training of large language models (LLMs). While DPO improves model guardrails against such attacks, we show that the efficacy of refusal learning varies drastically depending on the model's original post-training alignment scheme--e.g., GRPO-based LLMs learn to refuse extremely poorly. Thus, to further improve FBA refusals, we introduce Retrieval Augmented Generation for Preference alignment (RAG-Pref), a novel preference alignment strategy based on RAG. We show that RAG-Pref significantly improves the ability of LLMs to refuse FBAs, particularly when combined with DPO alignment, thus drastically improving guardrails against MCP-based attacks.

摘要: 模型上下文协议（HCP）已被广泛采用为开放标准，实现生成性人工智能代理的无缝集成。然而，最近的工作表明，HCP很容易受到基于检索的“错误良性”攻击（FBA），允许恶意系统访问和凭证盗窃，但要求用户将受损文件直接下载到其系统。在此，我们表明基于MPP的攻击的威胁模型比之前想象的要广泛得多，即攻击者只需在网上发布恶意内容就可以欺骗HCP代理对毫无戒心的受害者系统实施攻击。   为了改善针对此类攻击的对齐护栏，我们引入了一个由FBA和（真正）良性样本组成的新的HCP数据集，以探索直接偏好优化（DPO）用于大型语言模型（LLM）拒绝训练的有效性。虽然DPO改善了模型针对此类攻击的护栏，但我们表明拒绝学习的功效根据模型原始的训练后对齐方案而变化很大--例如，基于GRPO的LLM拒绝能力极差。因此，为了进一步改善FBA拒绝，我们引入了偏好对齐检索增强生成（RAG-Pref），这是一种基于RAG的新型偏好对齐策略。我们表明，RAG-Pref显着提高了LLM拒绝FBA的能力，特别是与DPO对齐相结合时，从而大大改善了针对基于MPP的攻击的护栏。



## **36. Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models**

合并劫持：对大型语言模型合并的后门攻击 cs.CR

This paper is accepted by ACL 2025 main conference

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23561v1) [paper-pdf](http://arxiv.org/pdf/2505.23561v1)

**Authors**: Zenghui Yuan, Yangming Xu, Jiawen Shi, Pan Zhou, Lichao Sun

**Abstract**: Model merging for Large Language Models (LLMs) directly fuses the parameters of different models finetuned on various tasks, creating a unified model for multi-domain tasks. However, due to potential vulnerabilities in models available on open-source platforms, model merging is susceptible to backdoor attacks. In this paper, we propose Merge Hijacking, the first backdoor attack targeting model merging in LLMs. The attacker constructs a malicious upload model and releases it. Once a victim user merges it with any other models, the resulting merged model inherits the backdoor while maintaining utility across tasks. Merge Hijacking defines two main objectives-effectiveness and utility-and achieves them through four steps. Extensive experiments demonstrate the effectiveness of our attack across different models, merging algorithms, and tasks. Additionally, we show that the attack remains effective even when merging real-world models. Moreover, our attack demonstrates robustness against two inference-time defenses (Paraphrasing and CLEANGEN) and one training-time defense (Fine-pruning).

摘要: 大型语言模型（LLM）的模型合并直接融合对各种任务进行微调的不同模型的参数，为多领域任务创建统一模型。然而，由于开源平台上可用的模型存在潜在漏洞，模型合并很容易受到后门攻击。本文提出了合并劫持，这是LLM中第一个针对合并的后门攻击模型。攻击者构建恶意上传模型并将其发布。一旦受害用户将其与任何其他模型合并，生成的合并模型将继承后门，同时保持跨任务的实用性。合并劫持定义了两个主要目标--有效性和实用性--并通过四个步骤实现它们。大量实验证明了我们在不同模型、合并算法和任务中的攻击的有效性。此外，我们表明，即使在合并现实世界模型时，攻击仍然有效。此外，我们的攻击表现出了对两种推理时防御（Paraphrapping和CleangEN）和一种训练时防御（Fine-修剪）的鲁棒性。



## **37. SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents**

安全科学家：LLM代理人的风险意识科学发现 cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23559v1) [paper-pdf](http://arxiv.org/pdf/2505.23559v1)

**Authors**: Kunlun Zhu, Jiaxun Zhang, Ziheng Qi, Nuoxing Shang, Zijia Liu, Peixuan Han, Yue Su, Haofei Yu, Jiaxuan You

**Abstract**: Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce \textbf{SafeScientist}, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tool-use monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35\% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at https://github.com/ulab-uiuc/SafeScientist. \textcolor{red}{Warning: this paper contains example data that may be offensive or harmful.}

摘要: 大型语言模型（LLM）代理的最新进展显着加速了科学发现自动化，但同时提出了关键的伦理和安全问题。为了系统地应对这些挑战，我们引入了一个创新的人工智能科学家框架，旨在增强人工智能驱动的科学探索中的安全和道德责任。SafeScientist主动拒绝道德上不合适或高风险的任务，并在整个研究过程中严格强调安全。为了实现全面的安全监督，我们整合了多种防御机制，包括即时监控、代理协作监控、工具使用监控和道德审查员组件。作为SafeScientist的补充，我们提出了\textBF{SciSafetyBench}，这是一个专门用于评估科学背景下人工智能安全性的新型基准，包括6个领域的240项高风险科学任务，以及30个专门设计的科学工具和120个工具相关的风险任务。大量实验表明，与传统的人工智能科学家框架相比，SafeScientist将安全性能显着提高了35%，而不会影响科学输出质量。此外，我们还严格验证了我们的安全管道针对各种对抗攻击方法的稳健性，进一步证实了我们集成方法的有效性。代码和数据可在https://github.com/ulab-uiuc/SafeScientist上获取。\textcolor{red}{警告：本文包含可能令人反感或有害的示例数据。}



## **38. Hijacking Large Language Models via Adversarial In-Context Learning**

通过对抗性上下文学习劫持大型语言模型 cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2311.09948v3) [paper-pdf](http://arxiv.org/pdf/2311.09948v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Prashant Khanduri, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the preconditioned prompts. Despite its promising performance, crafted adversarial attacks pose a notable threat to the robustness of LLMs. Existing attacks are either easy to detect, require a trigger in user input, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable prompt injection attack against ICL, aiming to hijack LLMs to generate the target output or elicit harmful responses. In our threat model, the hacker acts as a model publisher who leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos via prompt injection. We also propose effective defense strategies using a few shots of clean demos, enhancing the robustness of LLMs during ICL. Extensive experimental results across various classification and jailbreak tasks demonstrate the effectiveness of the proposed attack and defense strategies. This work highlights the significant security vulnerabilities of LLMs during ICL and underscores the need for further in-depth studies.

摘要: 上下文学习（ICL）已成为一种强大的范式，通过利用带标签的示例作为预处理提示中的演示（演示），利用LLM来执行特定的下游任务。尽管性能令人鼓舞，但精心设计的对抗攻击对LLM的稳健性构成了显着的威胁。现有的攻击要么容易检测，需要用户输入触发，要么缺乏针对ICL的特异性。为了解决这些问题，这项工作引入了一种针对ICL的新型可转移即时注入攻击，旨在劫持LLM以生成目标输出或引发有害响应。在我们的威胁模型中，黑客充当模型发布者，利用基于梯度的提示搜索方法来学习难以察觉的对抗性后缀，并通过提示注入将其添加到上下文演示中。我们还使用几次干净的演示提出了有效的防御策略，增强ICL期间LLM的稳健性。各种分类和越狱任务的大量实验结果证明了所提出的攻击和防御策略的有效性。这项工作强调了ICL期间LLM的重大安全漏洞，并强调了进一步深入研究的必要性。



## **39. Learning to Poison Large Language Models for Downstream Manipulation**

学习毒害大型语言模型以进行下游操作 cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2402.13459v3) [paper-pdf](http://arxiv.org/pdf/2402.13459v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Mohammad Amin Roshani, Prashant Khanduri, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where the adversary inserts backdoor triggers into training data to manipulate outputs. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the supervised fine-tuning (SFT) process. We propose a novel gradient-guided backdoor trigger learning (GBTL) algorithm to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various language model tasks, including sentiment analysis, domain generation, and question answering, our poisoning strategy demonstrates a high success rate in compromising various LLMs' outputs. We further propose two defense strategies against data poisoning attacks, including in-context learning (ICL) and continuous learning (CL), which effectively rectify the behavior of LLMs and significantly reduce the decline in performance. Our work highlights the significant security risks present during SFT of LLMs and the necessity of safeguarding LLMs against data poisoning attacks.

摘要: 大型语言模型（LLM）的出现标志着语言处理和推理能力取得了重大成就。尽管LLM取得了进步，但仍面临数据中毒攻击的漏洞，即对手将后门触发器插入训练数据中以操纵输出。这项工作通过设计专门针对利用监督式微调（SFT）过程而定制的新数据中毒攻击，进一步识别了LLM中的额外安全风险。我们提出了一种新型的梯度引导后门触发学习（GBTL）算法来有效识别对抗触发，确保逃避传统防御的检测，同时保持内容完整性。通过对各种语言模型任务（包括情感分析、领域生成和问题回答）的实验验证，我们的中毒策略证明了损害各种LLM输出的高成功率。我们进一步提出了两种针对数据中毒攻击的防御策略，包括上下文学习（ICL）和持续学习（CL），有效纠正LLM的行为，显着减少性能下降。我们的工作强调了LLM SFT期间存在的重大安全风险以及保护LLM免受数据中毒攻击的必要性。



## **40. Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models**

分而治之：击败多模式大型语言模型的混合策略 cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2412.16555v3) [paper-pdf](http://arxiv.org/pdf/2412.16555v3)

**Authors**: Yanxu Mao, Peipei Liu, Tiehan Cui, Zhaoteng Yan, Congying Liu, Datao You

**Abstract**: Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive dataset for multimodal jailbreaking research: TriJail, which includes jailbreak prompts for all three modalities. Experiments on the TriJail dataset and the benchmark dataset AdvBench, conducted on 13 popular LLMs, demonstrate advanced attack success rates and significant reduction in time overhead.

摘要: 大型语言模型（LLM）因其强大的推理、理解和生成能力而广泛应用于社会各个领域。然而，与这些模型相关的安全问题正变得日益严重。越狱攻击作为检测LLM漏洞的重要方法，已被研究人员探索，他们试图通过各种攻击方法诱导这些模型生成有害内容。然而，现有的越狱方法面临着许多局限性，例如过多的查询次数、越狱模式的覆盖范围有限、攻击成功率低以及评估方法简单化。为了克服这些限制，本文提出了一种多模式越狱方法：JMLLM。该方法集成了多种策略，以跨文本、视觉和听觉方式执行全面的越狱攻击。此外，我们还为多模式越狱研究提供了一个新的全面数据集：TriJail，其中包括所有三种模式的越狱提示。在TriJail数据集和基准数据集AdvBench上进行的实验在13个流行的LLM上进行，展示了先进的攻击成功率和显着减少的时间成本。



## **41. DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing**

德尔曼：通过模型编辑对大型语言模型越狱的动态防御 cs.CR

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.11647v2) [paper-pdf](http://arxiv.org/pdf/2502.11647v2)

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection.

摘要: 大型语言模型（LLM）广泛应用于决策制定，但其部署受到越狱攻击的威胁，即敌对用户操纵模型行为以绕过安全措施。现有的防御机制，例如安全微调和模型编辑，要么需要大量的参数修改，要么缺乏精确性，导致一般任务的性能下降，不适合部署后的安全对齐。为了应对这些挑战，我们提出了DELMAN（LLC动态编辑JAilbreak DefeNse），这是一种利用直接模型编辑来精确、动态地保护免受越狱攻击的新颖方法。德尔曼直接更新最少的相关参数集，以中和有害行为，同时保留模型的实用性。为了避免在良性上下文中触发安全响应，我们引入了KL分歧正规化，以确保更新后的模型在处理良性查询时与原始模型保持一致。实验结果表明，DELMAN在缓解越狱攻击的同时保持模型的实用性方面优于基线方法，并无缝适应新的攻击实例，为部署后模型保护提供了实用高效的解决方案。



## **42. Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models**

基于大型语言模型语义理解能力的自适应越狱策略 cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23404v1) [paper-pdf](http://arxiv.org/pdf/2505.23404v1)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin

**Abstract**: Adversarial attacks on Large Language Models (LLMs) via jailbreaking techniques-methods that circumvent their built-in safety and ethical constraints-have emerged as a critical challenge in AI security. These attacks compromise the reliability of LLMs by exploiting inherent weaknesses in their comprehension capabilities. This paper investigates the efficacy of jailbreaking strategies that are specifically adapted to the diverse levels of understanding exhibited by different LLMs. We propose the Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models, a novel framework that classifies LLMs into Type I and Type II categories according to their semantic comprehension abilities. For each category, we design tailored jailbreaking strategies aimed at leveraging their vulnerabilities to facilitate successful attacks. Extensive experiments conducted on multiple LLMs demonstrate that our adaptive strategy markedly improves the success rate of jailbreaking. Notably, our approach achieves an exceptional 98.9% success rate in jailbreaking GPT-4o(29 May 2025 release)

摘要: 通过越狱技术（规避其内置安全和道德约束的方法）对大型语言模型（LLM）进行的对抗攻击已成为人工智能安全领域的一个关键挑战。这些攻击通过利用LLM理解能力的固有弱点来损害LLM的可靠性。本文研究了专门适应不同法学硕士所表现出的不同理解水平的越狱策略的有效性。我们提出了基于大型语言模型语义理解能力的自适应越狱策略，这是一个新颖的框架，根据它们的语义理解能力将LLM分为类型I和类型II类别。对于每个类别，我们设计了量身定制的越狱策略，旨在利用其漏洞来促进成功的攻击。在多个LLM上进行的广泛实验表明，我们的自适应策略显着提高了越狱的成功率。值得注意的是，我们的方法在越狱GPT-4 o（2025年5月29日发布）中实现了98.9%的卓越成功率



## **43. Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction**

数据集特征化：通过无监督数据重建揭示自然语言特征 cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.17541v2) [paper-pdf](http://arxiv.org/pdf/2502.17541v2)

**Authors**: Michal Bravansky, Vaclav Kubon, Suhas Hariharan, Robert Kirk

**Abstract**: Interpreting data is central to modern research. Large language models (LLMs) show promise in providing such natural language interpretations of data, yet simple feature extraction methods such as prompting often fail to produce accurate and versatile descriptions for diverse datasets and lack control over granularity and scale. To address these limitations, we propose a domain-agnostic method for dataset featurization that provides precise control over the number of features extracted while maintaining compact and descriptive representations comparable to human labeling. Our method optimizes the selection of informative binary features by evaluating the ability of an LLM to reconstruct the original data using those features. We demonstrate its effectiveness in dataset modeling tasks and through two case studies: (1) Constructing a feature representation of jailbreak tactics that compactly captures both the effectiveness and diversity of a larger set of human-crafted attacks; and (2) automating the discovery of features that align with human preferences, achieving accuracy and robustness comparable to human-crafted features. Moreover, we show that the pipeline scales effectively, improving as additional features are sampled, making it suitable for large and diverse datasets.

摘要: 解释数据是现代研究的核心。大型语言模型（LLM）在提供此类数据自然语言解释方面表现出希望，但提示等简单的特征提取方法往往无法为不同数据集生成准确且通用的描述，并且缺乏对粒度和规模的控制。为了解决这些限制，我们提出了一种用于数据集特征化的领域不可知方法，该方法可以精确控制提取的特征数量，同时保持与人类标记相当的紧凑和描述性表示。我们的方法通过评估LLM使用这些特征重建原始数据的能力来优化信息二进制特征的选择。我们通过两个案例研究证明了它在数据集建模任务中的有效性：（1）构建越狱策略的特征表示，该特征表示可以完整地捕获更大的人为攻击的有效性和多样性;（2）自动发现符合人类偏好的特征，实现与人为特征相媲美的准确性和鲁棒性。此外，我们还证明了流水线可以有效地扩展，随着额外特征的采样而改进，使其适用于大型和多样化的数据集。



## **44. Disrupting Vision-Language Model-Driven Navigation Services via Adversarial Object Fusion**

通过对抗对象融合扰乱视觉语言模型驱动的导航服务 cs.CR

Under review

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23266v1) [paper-pdf](http://arxiv.org/pdf/2505.23266v1)

**Authors**: Chunlong Xie, Jialing He, Shangwei Guo, Jiacheng Wang, Shudong Zhang, Tianwei Zhang, Tao Xiang

**Abstract**: We present Adversarial Object Fusion (AdvOF), a novel attack framework targeting vision-and-language navigation (VLN) agents in service-oriented environments by generating adversarial 3D objects. While foundational models like Large Language Models (LLMs) and Vision Language Models (VLMs) have enhanced service-oriented navigation systems through improved perception and decision-making, their integration introduces vulnerabilities in mission-critical service workflows. Existing adversarial attacks fail to address service computing contexts, where reliability and quality-of-service (QoS) are paramount. We utilize AdvOF to investigate and explore the impact of adversarial environments on the VLM-based perception module of VLN agents. In particular, AdvOF first precisely aggregates and aligns the victim object positions in both 2D and 3D space, defining and rendering adversarial objects. Then, we collaboratively optimize the adversarial object with regularization between the adversarial and victim object across physical properties and VLM perceptions. Through assigning importance weights to varying views, the optimization is processed stably and multi-viewedly by iterative fusions from local updates and justifications. Our extensive evaluations demonstrate AdvOF can effectively degrade agent performance under adversarial conditions while maintaining minimal interference with normal navigation tasks. This work advances the understanding of service security in VLM-powered navigation systems, providing computational foundations for robust service composition in physical-world deployments.

摘要: 我们提出了对抗性对象融合（AdvOF），这是一种新型攻击框架，通过生成对抗性3D对象，针对面向服务环境中的视觉和语言导航（VLN）代理。虽然大型语言模型（LLM）和视觉语言模型（VLM）等基础模型通过改进感知和决策增强了面向服务的导航系统，但它们的集成在关键任务服务工作流程中引入了漏洞。现有的对抗性攻击无法解决服务计算上下文，而服务计算上下文的可靠性和服务质量（Qos）至关重要。我们利用AdvOF来调查和探索对抗环境对VLN代理基于LM的感知模块的影响。特别是，AdvOF首先在2D和3D空间中精确地聚合和对齐受害对象的位置，定义和渲染对抗对象。然后，我们在物理属性和VLM感知之间通过对抗对象和受害者对象之间的正则化协作优化对抗对象。通过对不同视图赋予重要性权值，通过局部更新和调整的迭代融合，实现了稳定的多视图优化。我们的广泛评估表明，AdvOF可以有效地降低代理性能在对抗条件下，同时保持最小的干扰与正常的导航任务。这项工作促进了对基于VLM的导航系统中服务安全性的理解，为物理世界部署中的稳健服务组合提供了计算基础。



## **45. Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking**

推理防御：安全意识推理可以保护大型语言模型免受越狱 cs.CL

18 pages

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.12970v2) [paper-pdf](http://arxiv.org/pdf/2502.12970v2)

**Authors**: Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, Lei Sha

**Abstract**: Large Reasoning Models (LRMs) have demonstrated impressive performances across diverse domains. However, how safety of Large Language Models (LLMs) benefits from enhanced reasoning capabilities against jailbreak queries remains unexplored. To bridge this gap, in this paper, we propose Reasoning-to-Defend (R2D), a novel training paradigm that integrates a safety-aware reasoning mechanism into LLMs' generation. This enables self-evaluation at each step of the reasoning process, forming safety pivot tokens as indicators of the safety status of responses. Furthermore, in order to improve the accuracy of predicting pivot tokens, we propose Contrastive Pivot Optimization (CPO), which enhances the model's perception of the safety status of given dialogues. LLMs dynamically adjust their response strategies during reasoning, significantly enhancing their safety capabilities defending jailbreak attacks. Extensive experiments demonstrate that R2D effectively mitigates various attacks and improves overall safety, while maintaining the original performances. This highlights the substantial potential of safety-aware reasoning in improving robustness of LRMs and LLMs against various jailbreaks.

摘要: 大型推理模型（LRM）在不同领域表现出令人印象深刻的性能。然而，大型语言模型（LLM）的安全性如何从针对越狱查询的增强推理能力中受益仍有待探索。为了弥合这一差距，在本文中，我们提出了推理防御（R2 D），这是一种新型训练范式，将安全感知推理机制集成到LLM的生成中。这使得推理过程的每个步骤都能够进行自我评估，形成安全支点令牌作为响应安全状态的指标。此外，为了提高预测枢轴令牌的准确性，我们提出了对比枢轴优化（CPO），这增强了模型对给定对话的安全状态的感知。LLM在推理过程中动态调整其响应策略，显著增强其防御越狱攻击的安全能力。大量的实验表明，R2D有效地减轻了各种攻击，提高了整体安全性，同时保持了原有的性能。这突出了安全意识推理在提高LRM和LLM对各种越狱的鲁棒性方面的巨大潜力。



## **46. Jailbreaking to Jailbreak**

越狱到越狱 cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.09638v2) [paper-pdf](http://arxiv.org/pdf/2502.09638v2)

**Authors**: Jeremy Kritz, Vaughn Robinson, Robert Vacareanu, Bijan Varjavand, Michael Choi, Bobby Gogov, Scale Red Team, Summer Yue, Willow E. Primack, Zifan Wang

**Abstract**: Large Language Models (LLMs) can be used to red team other models (e.g. jailbreaking) to elicit harmful contents. While prior works commonly employ open-weight models or private uncensored models for doing jailbreaking, as the refusal-training of strong LLMs (e.g. OpenAI o3) refuse to help jailbreaking, our work turn (almost) any black-box LLMs into attackers. The resulting $J_2$ (jailbreaking-to-jailbreak) attackers can effectively jailbreak the safeguard of target models using various strategies, both created by themselves or from expert human red teamers. In doing so, we show their strong but under-researched jailbreaking capabilities. Our experiments demonstrate that 1) prompts used to create $J_2$ attackers transfer across almost all black-box models; 2) an $J_2$ attacker can jailbreak a copy of itself, and this vulnerability develops rapidly over the past 12 months; 3) reasong models, such as Sonnet-3.7, are strong $J_2$ attackers compared to others. For example, when used against the safeguard of GPT-4o, $J_2$ (Sonnet-3.7) achieves 0.975 attack success rate (ASR), which matches expert human red teamers and surpasses the state-of-the-art algorithm-based attacks. Among $J_2$ attackers, $J_2$ (o3) achieves highest ASR (0.605) against Sonnet-3.5, one of the most robust models.

摘要: 大型语言模型（LLM）可用于与其他模型（例如越狱）进行团队合作，以引出有害内容。虽然之前的作品通常使用开放权重模型或私人未经审查的模型来进行越狱，但由于强大的LLM（例如OpenAI o3）的重新训练拒绝帮助越狱，我们的工作将（几乎）任何黑匣子LLM变成攻击者。由此产生的$J_2$（越狱到越狱）攻击者可以使用各种策略有效地越狱目标模型的保护，无论是由他们自己创建的还是由专家人类红色团队创建的。通过这样做，我们展示了他们强大但研究不足的越狱能力。我们的实验表明：1）用于创建$J_2$攻击者在几乎所有黑匣子模型中传输的提示; 2）$J_2$攻击者可以越狱自己的副本，并且该漏洞在过去12个月内迅速发展; 3）Reason模型，例如Sonnet-3.7，与其他模型相比，是强大的$J_2$攻击者。例如，当针对GPT-4 o的保护使用时，$J_2$（Sonnet-3.7）可实现0.975的攻击成功率（ASB），这与专家人类红队队员相匹配，并超越了最先进的基于算法的攻击。在$J_2$攻击者中，$J_2$（o3）针对Sonnet-3.5（最强大的模型之一）实现了最高的ASB（0.605）。



## **47. OMNIGUARD: An Efficient Approach for AI Safety Moderation Across Modalities**

OMNIGUARD：跨模式人工智能安全调节的有效方法 cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23856v1) [paper-pdf](http://arxiv.org/pdf/2505.23856v1)

**Authors**: Sahil Verma, Keegan Hines, Jeff Bilmes, Charlotte Siska, Luke Zettlemoyer, Hila Gonen, Chandan Singh

**Abstract**: The emerging capabilities of large language models (LLMs) have sparked concerns about their immediate potential for harmful misuse. The core approach to mitigate these concerns is the detection of harmful queries to the model. Current detection approaches are fallible, and are particularly susceptible to attacks that exploit mismatched generalization of model capabilities (e.g., prompts in low-resource languages or prompts provided in non-text modalities such as image and audio). To tackle this challenge, we propose OMNIGUARD, an approach for detecting harmful prompts across languages and modalities. Our approach (i) identifies internal representations of an LLM/MLLM that are aligned across languages or modalities and then (ii) uses them to build a language-agnostic or modality-agnostic classifier for detecting harmful prompts. OMNIGUARD improves harmful prompt classification accuracy by 11.57\% over the strongest baseline in a multilingual setting, by 20.44\% for image-based prompts, and sets a new SOTA for audio-based prompts. By repurposing embeddings computed during generation, OMNIGUARD is also very efficient ($\approx 120 \times$ faster than the next fastest baseline). Code and data are available at: https://github.com/vsahil/OmniGuard.

摘要: 大型语言模型（LLM）的新兴功能引发了人们对其直接潜在有害滥用的担忧。缓解这些担忧的核心方法是检测对模型的有害查询。当前的检测方法是容易出错的，并且特别容易受到利用模型能力不匹配的概括的攻击（例如，低资源语言的提示或以图像和音频等非文本形式提供的提示）。为了应对这一挑战，我们提出了OMNIGUARD，这是一种检测跨语言和模式有害提示的方法。我们的方法（i）识别跨语言或模式对齐的LLM/MLLM的内部表示，然后（ii）使用它们来构建语言不可知或模式不可知的分类器，用于检测有害提示。OMNIGUARD将有害提示分类准确性比多语言设置中的最强基线提高了11.57%，对于基于图像的提示提高了20.44%，并为基于音频的提示设置了新的SOTA。通过重新利用生成期间计算的嵌入，OMNIGUARD也非常高效（比下一个最快的基线快$\大约120 \乘以$）。代码和数据可访问：https://github.com/vsahil/OmniGuard。



## **48. DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors**

DyePack：使用后门可证明标记LLM中的测试集污染 cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23001v1) [paper-pdf](http://arxiv.org/pdf/2505.23001v1)

**Authors**: Yize Cheng, Wenxiao Wang, Mazda Moayeri, Soheil Feizi

**Abstract**: Open benchmarks are essential for evaluating and advancing large language models, offering reproducibility and transparency. However, their accessibility makes them likely targets of test set contamination. In this work, we introduce DyePack, a framework that leverages backdoor attacks to identify models that used benchmark test sets during training, without requiring access to the loss, logits, or any internal details of the model. Like how banks mix dye packs with their money to mark robbers, DyePack mixes backdoor samples with the test data to flag models that trained on it. We propose a principled design incorporating multiple backdoors with stochastic targets, enabling exact false positive rate (FPR) computation when flagging every model. This provably prevents false accusations while providing strong evidence for every detected case of contamination. We evaluate DyePack on five models across three datasets, covering both multiple-choice and open-ended generation tasks. For multiple-choice questions, it successfully detects all contaminated models with guaranteed FPRs as low as 0.000073% on MMLU-Pro and 0.000017% on Big-Bench-Hard using eight backdoors. For open-ended generation tasks, it generalizes well and identifies all contaminated models on Alpaca with a guaranteed false positive rate of just 0.127% using six backdoors.

摘要: 开放基准对于评估和推进大型语言模型、提供可重复性和透明度至关重要。然而，它们的可及性使它们可能成为测试集污染的目标。在这项工作中，我们引入了DyePack，这是一个利用后门攻击来识别在训练期间使用基准测试集的模型的框架，而不需要访问模型的损失、日志或任何内部细节。就像银行将染料包与钱混合来标记劫匪一样，DyePack将后门样本与测试数据混合起来，以标记对其进行训练的模型。我们提出了一种原则性设计，将多个后门与随机目标结合在一起，在标记每个模型时实现精确的假阳性率（FPR）计算。事实证明，这可以防止虚假指控，同时为每一个检测到的污染案例提供强有力的证据。我们在三个数据集的五个模型上评估了DyePack，涵盖多项选择和开放式生成任务。对于多项选择题，它使用八个后门成功检测到所有受污染的型号，保证FPR在MMLU-Pro上低至0.000073%，在Big-Bench-Hard上低至0.00017%。对于开放式生成任务，它可以很好地推广，并使用六个后门识别羊驼上所有受污染的模型，保证假阳性率仅为0.127%。



## **49. Revisiting Multi-Agent Debate as Test-Time Scaling: A Systematic Study of Conditional Effectiveness**

重新审视多主体辩论作为测试时间缩放：条件有效性的系统性研究 cs.AI

Preprint, under review

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.22960v1) [paper-pdf](http://arxiv.org/pdf/2505.22960v1)

**Authors**: Yongjin Yang, Euiin Yi, Jongwoo Ko, Kimin Lee, Zhijing Jin, Se-Young Yun

**Abstract**: The remarkable growth in large language model (LLM) capabilities has spurred exploration into multi-agent systems, with debate frameworks emerging as a promising avenue for enhanced problem-solving. These multi-agent debate (MAD) approaches, where agents collaboratively present, critique, and refine arguments, potentially offer improved reasoning, robustness, and diverse perspectives over monolithic models. Despite prior studies leveraging MAD, a systematic understanding of its effectiveness compared to self-agent methods, particularly under varying conditions, remains elusive. This paper seeks to fill this gap by conceptualizing MAD as a test-time computational scaling technique, distinguished by collaborative refinement and diverse exploration capabilities. We conduct a comprehensive empirical investigation comparing MAD with strong self-agent test-time scaling baselines on mathematical reasoning and safety-related tasks. Our study systematically examines the influence of task difficulty, model scale, and agent diversity on MAD's performance. Key findings reveal that, for mathematical reasoning, MAD offers limited advantages over self-agent scaling but becomes more effective with increased problem difficulty and decreased model capability, while agent diversity shows little benefit. Conversely, for safety tasks, MAD's collaborative refinement can increase vulnerability, but incorporating diverse agent configurations facilitates a gradual reduction in attack success through the collaborative refinement process. We believe our findings provide critical guidance for the future development of more effective and strategically deployed MAD systems.

摘要: 大型语言模型（LLM）能力的显着增长刺激了对多代理系统的探索，辩论框架成为增强问题解决的有希望的途径。这些多主体辩论（MAD）方法，主体协作地呈现、批评和完善论点，与单一模型相比，有可能提供更好的推理、稳健性和多样化的观点。尽管之前的研究利用了MAD，但系统地了解其与自代理方法相比的有效性，特别是在不同条件下，仍然难以捉摸。本文试图通过将MAD概念化为一种测试时计算缩放技术来填补这一空白，该技术的特点是协作细化和多样化的探索能力。我们进行了一项全面的实证研究，将MAD与数学推理和安全相关任务的强大自代理测试时间缩放基线进行了比较。我们的研究系统地考察了任务难度、模型规模和代理多样性对MAD绩效的影响。主要研究结果表明，对于数学推理，MAD比自代理扩展提供的优势有限，但随着问题难度的增加和模型能力的降低而变得更加有效，而代理多样性几乎没有表现出什么好处。相反，对于安全任务，MAD的协作细化可能会增加脆弱性，但结合不同的代理配置有助于通过协作细化过程逐渐降低攻击成功率。我们相信，我们的研究结果为未来开发更有效和战略部署的MAD系统提供了重要指导。



## **50. Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates**

LLM可以欺骗CLIP吗？通过文本更新对预训练多模式表示的对抗性组合进行基准测试 cs.CL

ACL 2025 Main. Code is released at  https://vision.snu.ac.kr/projects/mac

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22943v1) [paper-pdf](http://arxiv.org/pdf/2505.22943v1)

**Authors**: Jaewoo Ahn, Heeseung Yun, Dayoon Ko, Gunhee Kim

**Abstract**: While pre-trained multimodal representations (e.g., CLIP) have shown impressive capabilities, they exhibit significant compositional vulnerabilities leading to counterintuitive judgments. We introduce Multimodal Adversarial Compositionality (MAC), a benchmark that leverages large language models (LLMs) to generate deceptive text samples to exploit these vulnerabilities across different modalities and evaluates them through both sample-wise attack success rate and group-wise entropy-based diversity. To improve zero-shot methods, we propose a self-training approach that leverages rejection-sampling fine-tuning with diversity-promoting filtering, which enhances both attack success rate and sample diversity. Using smaller language models like Llama-3.1-8B, our approach demonstrates superior performance in revealing compositional vulnerabilities across various multimodal representations, including images, videos, and audios.

摘要: 虽然预训练的多模式表示（例如，CLIP）表现出令人印象深刻的能力，它们表现出显着的合成漏洞，导致反直觉的判断。我们引入了多模式对抗组合（MAC），这是一个基准，利用大型语言模型（LLM）来生成欺骗性文本样本，以利用不同模式中的这些漏洞，并通过样本攻击成功率和基于分组的基于信息的多样性来评估它们。为了改进零射击方法，我们提出了一种自训练方法，该方法利用拒绝采样微调和促进多样性的过滤，从而增强了攻击成功率和样本多样性。使用Llama-3.1-8B等较小的语言模型，我们的方法在揭示各种多模式表示（包括图像、视频和音频）的合成漏洞方面表现出卓越的性能。



