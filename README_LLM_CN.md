# Latest Large Language Model Attack Papers
**update at 2025-11-07 11:06:50**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Large Language Models for Cyber Security**

网络安全的大型语言模型 cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04508v1) [paper-pdf](http://arxiv.org/pdf/2511.04508v1)

**Authors**: Raunak Somani, Aswani Kumar Cherukuri

**Abstract**: This paper studies the integration off Large Language Models into cybersecurity tools and protocols. The main issue discussed in this paper is how traditional rule-based and signature based security systems are not enough to deal with modern AI powered cyber threats. Cybersecurity industry is changing as threats are becoming more dangerous and adaptive in nature by levering the features provided by AI tools. By integrating LLMs into these tools and protocols, make the systems scalable, context-aware and intelligent. Thus helping it to mitigate these evolving cyber threats. The paper studies the architecture and functioning of LLMs, its integration into Encrypted prompts to prevent prompt injection attacks. It also studies the integration of LLMs into cybersecurity tools using a four layered architecture. At last, the paper has tried to explain various ways of integration LLMs into traditional Intrusion Detection System and enhancing its original abilities in various dimensions. The key findings of this paper has been (i)Encrypted Prompt with LLM is an effective way to mitigate prompt injection attacks, (ii) LLM enhanced cyber security tools are more accurate, scalable and adaptable to new threats as compared to traditional models, (iii) The decoupled model approach for LLM integration into IDS is the best way as it is the most accurate way.

摘要: 本文研究了大型语言模型与网络安全工具和协议的集成。本文讨论的主要问题是传统的基于规则和基于签名的安全系统如何不足以应对现代人工智能驱动的网络威胁。网络安全行业正在发生变化，因为威胁通过利用人工智能工具提供的功能变得更加危险和适应性。通过将LLM集成到这些工具和协议中，使系统具有可扩展性、上下文感知性和智能性。从而帮助其减轻这些不断变化的网络威胁。本文研究了LLM的架构和功能，以及将其集成到加密提示中以防止提示注入攻击。它还研究了使用四层架构将LLM集成到网络安全工具中的情况。最后，文章尝试解释了将LLM集成到传统入侵检测系统中并在各个维度上增强其原有能力的各种方法。本文的主要发现是（i）使用LLM的加密提示是减轻即时注入攻击的有效方法，（ii）与传统模型相比，LLM增强型网络安全工具更准确、可扩展且可适应新威胁，（iii）将LLM集成到IDS的脱钩模型方法是最好的方法，因为它是最准确的方法。



## **2. AdversariaLLM: A Unified and Modular Toolbox for LLM Robustness Research**

AdversariaLLM：用于LLM稳健性研究的统一模块化工作空间 cs.AI

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04316v1) [paper-pdf](http://arxiv.org/pdf/2511.04316v1)

**Authors**: Tim Beyer, Jonas Dornbusch, Jakob Steimle, Moritz Ladenburger, Leo Schwinn, Stephan Günnemann

**Abstract**: The rapid expansion of research on Large Language Model (LLM) safety and robustness has produced a fragmented and oftentimes buggy ecosystem of implementations, datasets, and evaluation methods. This fragmentation makes reproducibility and comparability across studies challenging, hindering meaningful progress. To address these issues, we introduce AdversariaLLM, a toolbox for conducting LLM jailbreak robustness research. Its design centers on reproducibility, correctness, and extensibility. The framework implements twelve adversarial attack algorithms, integrates seven benchmark datasets spanning harmfulness, over-refusal, and utility evaluation, and provides access to a wide range of open-weight LLMs via Hugging Face. The implementation includes advanced features for comparability and reproducibility such as compute-resource tracking, deterministic results, and distributional evaluation techniques. \name also integrates judging through the companion package JudgeZoo, which can also be used independently. Together, these components aim to establish a robust foundation for transparent, comparable, and reproducible research in LLM safety.

摘要: 对大型语言模型（LLM）安全性和稳健性研究的迅速扩展，产生了一个由实现、数据集和评估方法组成的碎片化且经常存在缺陷的生态系统。这种碎片化使得研究之间的重复性和可比性具有挑战性，阻碍了有意义的进展。为了解决这些问题，我们引入了AdversariaLLM，这是一个用于进行LLM越狱稳健性研究的工具箱。其设计以可重复性、正确性和可扩展性为中心。该框架实现了十二种对抗攻击算法，集成了涵盖危害性、过度拒绝和效用评估的七个基准数据集，并通过Hugging Face提供了对广泛开放权重LLM的访问权限。该实现包括可比性和可重复性的高级功能，例如计算资源跟踪、确定性结果和分布式评估技术。\Name还通过配套包JudgeZoo集成了判断，该包也可以独立使用。这些组成部分的目标是为LLM安全性的透明、可比和可重复的研究奠定坚实的基础。



## **3. Black-Box Guardrail Reverse-engineering Attack**

黑盒保护逆向工程攻击 cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04215v1) [paper-pdf](http://arxiv.org/pdf/2511.04215v1)

**Authors**: Hongwei Yao, Yun Xia, Shuo Shao, Haoran Shi, Tong Qiao, Cong Wang

**Abstract**: Large language models (LLMs) increasingly employ guardrails to enforce ethical, legal, and application-specific constraints on their outputs. While effective at mitigating harmful responses, these guardrails introduce a new class of vulnerabilities by exposing observable decision patterns. In this work, we present the first study of black-box LLM guardrail reverse-engineering attacks. We propose Guardrail Reverse-engineering Attack (GRA), a reinforcement learning-based framework that leverages genetic algorithm-driven data augmentation to approximate the decision-making policy of victim guardrails. By iteratively collecting input-output pairs, prioritizing divergence cases, and applying targeted mutations and crossovers, our method incrementally converges toward a high-fidelity surrogate of the victim guardrail. We evaluate GRA on three widely deployed commercial systems, namely ChatGPT, DeepSeek, and Qwen3, and demonstrate that it achieves an rule matching rate exceeding 0.92 while requiring less than $85 in API costs. These findings underscore the practical feasibility of guardrail extraction and highlight significant security risks for current LLM safety mechanisms. Our findings expose critical vulnerabilities in current guardrail designs and highlight the urgent need for more robust defense mechanisms in LLM deployment.

摘要: 大型语言模型（LLM）越来越多地采用护栏来对其输出实施道德、法律和特定应用程序的约束。虽然这些护栏可以有效减轻有害反应，但通过暴露可观察到的决策模式引入了一类新的漏洞。在这项工作中，我们首次研究了黑匣子LLM护栏反向工程攻击。我们提出了Guardious反向工程攻击（GRA），这是一个基于强化学习的框架，利用遗传算法驱动的数据增强来逼近受害者护栏的决策政策。通过迭代收集输入-输出对、优先考虑分歧情况以及应用有针对性的突变和交叉，我们的方法逐渐收敛到受害者护栏的高保真替代品。我们对三个广泛部署的商业系统（即ChatGPT、DeepSeek和Qwen 3）进行了GRA评估，并证明它的规则匹配率超过0.92，同时所需的API成本低于85美元。这些发现强调了护栏拆除的实际可行性，并强调了当前LLM安全机制的重大安全风险。我们的研究结果暴露了当前护栏设计中的关键漏洞，并强调了LLM部署中迫切需要更强大的防御机制。



## **4. Transferable & Stealthy Ensemble Attacks: A Black-Box Jailbreaking Framework for Large Language Models**

可转移和隐形的群体攻击：大型语言模型的黑匣子越狱框架 cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2410.23558v3) [paper-pdf](http://arxiv.org/pdf/2410.23558v3)

**Authors**: Yiqi Yang, Hongye Fu

**Abstract**: We present a novel black-box jailbreaking framework that integrates multiple LLM-as-Attacker strategies to deliver highly transferable and effective attacks. The framework is grounded in three key insights from prior jailbreaking research and practice: ensemble approaches outperform single methods in exposing aligned LLM vulnerabilities, malicious instructions vary in jailbreaking difficulty requiring tailored optimization, and disrupting semantic coherence of malicious prompts can manipulate their embeddings to boost success rates. Validated in the Competition for LLM and Agent Safety 2024, our solution achieved top rankings in the Jailbreaking Attack Track.

摘要: 我们提出了一种新颖的黑匣子越狱框架，该框架集成了多种LLM作为攻击者的策略，以提供高度可转移且有效的攻击。该框架基于之前越狱研究和实践的三个关键见解：集成方法在暴露一致的LLM漏洞方面优于单一方法，恶意指令在越狱难度方面各不相同，需要定制优化，破坏恶意提示的语义一致性可以操纵其嵌入以提高成功率。我们的解决方案在2024年法学硕士和代理安全竞赛中获得了验证，在越狱攻击赛道中获得了最高排名。



## **5. Measuring the Security of Mobile LLM Agents under Adversarial Prompts from Untrusted Third-Party Channels**

在来自不受信任的第三方渠道的对抗性承诺下衡量移动LLM代理的安全性 cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2510.27140v2) [paper-pdf](http://arxiv.org/pdf/2510.27140v2)

**Authors**: Chenghao Du, Quanfeng Huang, Tingxuan Tang, Zihao Wang, Adwait Nadkarni, Yue Xiao

**Abstract**: Large Language Models (LLMs) have transformed software development, enabling AI-powered applications known as LLM-based agents that promise to automate tasks across diverse apps and workflows. Yet, the security implications of deploying such agents in adversarial mobile environments remain poorly understood. In this paper, we present the first systematic study of security risks in mobile LLM agents. We design and evaluate a suite of adversarial case studies, ranging from opportunistic manipulations such as pop-up advertisements to advanced, end-to-end workflows involving malware installation and cross-app data exfiltration. Our evaluation covers eight state-of-the-art mobile agents across three architectures, with over 2,000 adversarial and paired benign trials. The results reveal systemic vulnerabilities: low-barrier vectors such as fraudulent ads succeed with over 80% reliability, while even workflows requiring the circumvention of operating-system warnings, such as malware installation, are consistently completed by advanced multi-app agents. By mapping these attacks to the MITRE ATT&CK Mobile framework, we uncover novel privilege-escalation and persistence pathways unique to LLM-driven automation. Collectively, our findings provide the first end-to-end evidence that mobile LLM agents are exploitable in realistic adversarial settings, where untrusted third-party channels (e.g., ads, embedded webviews, cross-app notifications) are an inherent part of the mobile ecosystem.

摘要: 大型语言模型（LLM）已经改变了软件开发，使AI驱动的应用程序成为基于LLM的代理，这些代理承诺在不同的应用程序和工作流中自动执行任务。然而，在对抗性移动环境中部署此类代理的安全影响仍然知之甚少。在本文中，我们提出了第一个系统的研究，在移动LLM代理的安全风险。我们设计和评估了一系列对抗性案例研究，从弹出式广告等机会主义操纵到涉及恶意软件安装和跨应用程序数据泄露的高级端到端工作流。我们的评估涵盖了三种架构中的八个最先进的移动代理，以及超过2，000项对抗性和配对良性试验。结果揭示了系统性漏洞：欺诈广告等低障碍载体成功，可靠性超过80%，而即使是需要规避操作系统警告的工作流程，例如恶意软件安装，也始终由高级多应用程序代理完成。通过将这些攻击映射到MITRE ATT & CK Mobile框架，我们发现了LLM驱动的自动化所独有的新型风险升级和持久性途径。总的来说，我们的研究结果提供了第一个端到端的证据，证明移动LLM代理在现实的对抗环境中是可利用的，其中不受信任的第三方渠道（例如，广告、嵌入式网络视图、跨应用程序通知）是移动生态系统的固有组成部分。



## **6. Whisper Leak: a side-channel attack on Large Language Models**

Whisper Leak：对大型语言模型的侧信道攻击 cs.CR

14 pages, 7 figures

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03675v1) [paper-pdf](http://arxiv.org/pdf/2511.03675v1)

**Authors**: Geoff McDonald, Jonathan Bar Or

**Abstract**: Large Language Models (LLMs) are increasingly deployed in sensitive domains including healthcare, legal services, and confidential communications, where privacy is paramount. This paper introduces Whisper Leak, a side-channel attack that infers user prompt topics from encrypted LLM traffic by analyzing packet size and timing patterns in streaming responses. Despite TLS encryption protecting content, these metadata patterns leak sufficient information to enable topic classification. We demonstrate the attack across 28 popular LLMs from major providers, achieving near-perfect classification (often >98% AUPRC) and high precision even at extreme class imbalance (10,000:1 noise-to-target ratio). For many models, we achieve 100% precision in identifying sensitive topics like "money laundering" while recovering 5-20% of target conversations. This industry-wide vulnerability poses significant risks for users under network surveillance by ISPs, governments, or local adversaries. We evaluate three mitigation strategies - random padding, token batching, and packet injection - finding that while each reduces attack effectiveness, none provides complete protection. Through responsible disclosure, we have collaborated with providers to implement initial countermeasures. Our findings underscore the need for LLM providers to address metadata leakage as AI systems handle increasingly sensitive information.

摘要: 大型语言模型（LLM）越来越多地部署在敏感领域，包括医疗保健、法律服务和保密通信，这些领域的隐私至关重要。本文介绍了Whisper Leak，这是一种侧通道攻击，通过分析流响应中的数据包大小和时间模式，从加密的LLM流量中推断出用户提示主题。尽管使用SSL加密保护内容，但这些元数据模式会泄露足够的信息来启用主题分类。我们展示了针对主要提供商的28种流行LLM的攻击，即使在极端类别失衡（10，000：1噪音与目标比）的情况下，也实现了近乎完美的分类（通常> 98%AUPRC）和高精度。对于许多模型，我们在识别“洗钱”等敏感话题方面实现了100%的准确度，同时恢复5 - 20%的目标对话。这种全行业的漏洞给受到ISP、政府或本地对手网络监视的用户带来了重大风险。我们评估了三种缓解策略--随机填充、令牌填充和数据包注入--发现虽然每种策略都会降低攻击有效性，但都没有提供完整的保护。通过负责任的披露，我们与提供商合作实施初步应对措施。我们的研究结果强调，随着人工智能系统处理日益敏感的信息，LLM提供商需要解决元数据泄露问题。



## **7. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2409.13174v4) [paper-pdf](http://arxiv.org/pdf/2409.13174v4)

**Authors**: Hao Cheng, Erjia Xiao, Yichi Wang, Chengyuan Yu, Mengshu Sun, Qiang Zhang, Jiahang Cao, Yijie Guo, Ning Liu, Kaidi Xu, Jize Zhang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.

摘要: 最近，在多模式大型语言模型（MLLM）进步的推动下，人们提出了视觉语言动作模型（VLAM），以在机器人操纵任务的开放词汇场景中实现更好的性能。由于操纵任务涉及与物理世界的直接互动，因此在执行该任务期间确保稳健性和安全性始终是一个非常关键的问题。本文通过综合当前对MLLM的安全性研究以及物理世界中操纵任务的具体应用场景，对VLAM在潜在物理威胁面前进行了全面评估。具体来说，我们提出了物理脆弱性评估管道（PVEP），它可以整合尽可能多的视觉模式物理威胁，以评估VLAM的物理稳健性。PVEP中的物理威胁具体包括分发外、基于印刷术的视觉提示和对抗性补丁攻击。通过比较VLAM受到攻击前后的性能波动，我们提供了VLAM如何响应不同物理威胁的可概括的\textBF{\textit{Analyses}。



## **8. Let the Bees Find the Weak Spots: A Path Planning Perspective on Multi-Turn Jailbreak Attacks against LLMs**

让蜜蜂找到弱点：针对LLM的多回合越狱攻击的路径规划视角 cs.CR

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03271v1) [paper-pdf](http://arxiv.org/pdf/2511.03271v1)

**Authors**: Yize Liu, Yunyun Hou, Aina Sui

**Abstract**: Large Language Models (LLMs) have been widely deployed across various applications, yet their potential security and ethical risks have raised increasing concerns. Existing research employs red teaming evaluations, utilizing multi-turn jailbreaks to identify potential vulnerabilities in LLMs. However, these approaches often lack exploration of successful dialogue trajectories within the attack space, and they tend to overlook the considerable overhead associated with the attack process. To address these limitations, this paper first introduces a theoretical model based on dynamically weighted graph topology, abstracting the multi-turn attack process as a path planning problem. Based on this framework, we propose ABC, an enhanced Artificial Bee Colony algorithm for multi-turn jailbreaks, featuring a collaborative search mechanism with employed, onlooker, and scout bees. This algorithm significantly improves the efficiency of optimal attack path search while substantially reducing the average number of queries required. Empirical evaluations on three open-source and two proprietary language models demonstrate the effectiveness of our approach, achieving attack success rates above 90\% across the board, with a peak of 98\% on GPT-3.5-Turbo, and outperforming existing baselines. Furthermore, it achieves comparable success with only 26 queries on average, significantly reducing red teaming overhead and highlighting its superior efficiency.

摘要: 大型语言模型（LLM）已广泛部署在各种应用程序中，但其潜在的安全和道德风险引起了越来越多的担忧。现有的研究采用红色团队评估，利用多回合越狱来识别LLM中的潜在漏洞。然而，这些方法通常缺乏对攻击空间内成功对话轨迹的探索，并且往往忽视与攻击过程相关的相当大的费用。为了解决这些局限性，本文首先引入了一个基于动态加权图布局的理论模型，将多回合攻击过程抽象为路径规划问题。基于这个框架，我们提出了ABC，这是一种用于多回合越狱的增强型人工蜂群算法，具有与受雇蜜蜂、旁观者和侦察蜜蜂的协作搜索机制。该算法显着提高了最佳攻击路径搜索的效率，同时大幅减少了所需的平均查询数量。对三种开源语言模型和两种专有语言模型的经验评估证明了我们方法的有效性，攻击成功率全面超过90%，GPT-3.5-Turbo的峰值为98%，并且优于现有基线。此外，它平均只需26个查询即可取得相当的成功，显着减少了红色团队管理费用，并凸显了其卓越的效率。



## **9. Death by a Thousand Prompts: Open Model Vulnerability Analysis**

千人死亡：开放模型漏洞分析 cs.CR

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03247v1) [paper-pdf](http://arxiv.org/pdf/2511.03247v1)

**Authors**: Amy Chang, Nicholas Conley, Harish Santhanalakshmi Ganesan, Adam Swanda

**Abstract**: Open-weight models provide researchers and developers with accessible foundations for diverse downstream applications. We tested the safety and security postures of eight open-weight large language models (LLMs) to identify vulnerabilities that may impact subsequent fine-tuning and deployment. Using automated adversarial testing, we measured each model's resilience against single-turn and multi-turn prompt injection and jailbreak attacks. Our findings reveal pervasive vulnerabilities across all tested models, with multi-turn attacks achieving success rates between 25.86\% and 92.78\% -- representing a $2\times$ to $10\times$ increase over single-turn baselines. These results underscore a systemic inability of current open-weight models to maintain safety guardrails across extended interactions. We assess that alignment strategies and lab priorities significantly influence resilience: capability-focused models such as Llama 3.3 and Qwen 3 demonstrate higher multi-turn susceptibility, whereas safety-oriented designs such as Google Gemma 3 exhibit more balanced performance.   The analysis concludes that open-weight models, while crucial for innovation, pose tangible operational and ethical risks when deployed without layered security controls. These findings are intended to inform practitioners and developers of the potential risks and the value of professional AI security solutions to mitigate exposure. Addressing multi-turn vulnerabilities is essential to ensure the safe, reliable, and responsible deployment of open-weight LLMs in enterprise and public domains. We recommend adopting a security-first design philosophy and layered protections to ensure resilient deployments of open-weight models.

摘要: 开放权重模型为研究人员和开发人员提供了各种下游应用程序的可用基础。我们测试了八个开放权重大型语言模型（LLM）的安全性和安全姿势，以识别可能影响后续微调和部署的漏洞。使用自动对抗测试，我们测量了每个模型对单回合和多回合提示注入和越狱攻击的弹性。我们的调查结果揭示了所有测试模型中普遍存在的漏洞，多回合攻击的成功率在25.86%和92.78%之间，比单回合基线增加了2美元到10美元。这些结果强调了当前开放重量模型系统性地无法在长期互动中维持安全护栏。我们评估了对齐策略和实验室优先事项对韧性有显着影响：Llama 3.3和Qwen 3等以能力为中心的模型表现出更高的多圈敏感性，而Google Gemma 3等以安全为导向的设计表现出更平衡的性能。   分析得出的结论是，开放权重模型虽然对创新至关重要，但在没有分层安全控制的情况下部署时会带来切实的运营和道德风险。这些调查结果旨在让从业者和开发人员了解专业人工智能安全解决方案的潜在风险和价值，以减轻风险。解决多回合漏洞对于确保在企业和公共领域安全、可靠和负责任地部署开放权重LLM至关重要。我们建议采用安全第一的设计理念和分层保护，以确保开重模型的弹性部署。



## **10. Adaptive and Robust Data Poisoning Detection and Sanitization in Wearable IoT Systems using Large Language Models**

使用大型语言模型在可穿戴物联网系统中进行自适应和稳健的数据中毒检测和清理 cs.LG

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02894v1) [paper-pdf](http://arxiv.org/pdf/2511.02894v1)

**Authors**: W. K. M Mithsara, Ning Yang, Ahmed Imteaj, Hussein Zangoti, Abdur R. Shahid

**Abstract**: The widespread integration of wearable sensing devices in Internet of Things (IoT) ecosystems, particularly in healthcare, smart homes, and industrial applications, has required robust human activity recognition (HAR) techniques to improve functionality and user experience. Although machine learning models have advanced HAR, they are increasingly susceptible to data poisoning attacks that compromise the data integrity and reliability of these systems. Conventional approaches to defending against such attacks often require extensive task-specific training with large, labeled datasets, which limits adaptability in dynamic IoT environments. This work proposes a novel framework that uses large language models (LLMs) to perform poisoning detection and sanitization in HAR systems, utilizing zero-shot, one-shot, and few-shot learning paradigms. Our approach incorporates \textit{role play} prompting, whereby the LLM assumes the role of expert to contextualize and evaluate sensor anomalies, and \textit{think step-by-step} reasoning, guiding the LLM to infer poisoning indicators in the raw sensor data and plausible clean alternatives. These strategies minimize reliance on curation of extensive datasets and enable robust, adaptable defense mechanisms in real-time. We perform an extensive evaluation of the framework, quantifying detection accuracy, sanitization quality, latency, and communication cost, thus demonstrating the practicality and effectiveness of LLMs in improving the security and reliability of wearable IoT systems.

摘要: 可穿戴传感设备在物联网（IoT）生态系统中的广泛集成，特别是在医疗保健、智能家居和工业应用中，需要强大的人类活动识别（HAR）技术来改善功能和用户体验。尽管机器学习模型具有高级HAR，但它们越来越容易受到数据中毒攻击，从而损害这些系统的数据完整性和可靠性。防御此类攻击的传统方法通常需要使用大型标记数据集进行广泛的任务特定训练，这限制了动态物联网环境中的适应性。这项工作提出了一种新颖的框架，该框架使用大型语言模型（LLM）在HAR系统中执行中毒检测和清理，利用零触发、单触发和少触发学习范式。我们的方法结合了\textit{role play}提示，LLM承担专家的角色来情境化和评估传感器异常，以及\textit{think分步}推理，指导LLM推断原始传感器数据中的中毒指标和合理的清洁替代品。这些策略最大限度地减少了对大量数据集管理的依赖，并实时实现强大、适应性强的防御机制。我们对框架进行了广泛的评估，量化检测准确性、消毒质量、延迟和通信成本，从而证明了LLM在提高可穿戴物联网系统安全性和可靠性方面的实用性和有效性。



## **11. Do Methods to Jailbreak and Defend LLMs Generalize Across Languages?**

越狱和捍卫LLM的方法是否适用于语言？ cs.CL

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.00689v2) [paper-pdf](http://arxiv.org/pdf/2511.00689v2)

**Authors**: Berk Atil, Rebecca J. Passonneau, Fred Morstatter

**Abstract**: Large language models (LLMs) undergo safety alignment after training and tuning, yet recent work shows that safety can be bypassed through jailbreak attacks. While many jailbreaks and defenses exist, their cross-lingual generalization remains underexplored. This paper presents the first systematic multilingual evaluation of jailbreaks and defenses across ten languages -- spanning high-, medium-, and low-resource languages -- using six LLMs on HarmBench and AdvBench. We assess two jailbreak types: logical-expression-based and adversarial-prompt-based. For both types, attack success and defense robustness vary across languages: high-resource languages are safer under standard queries but more vulnerable to adversarial ones. Simple defenses can be effective, but are language- and model-dependent. These findings call for language-aware and cross-lingual safety benchmarks for LLMs.

摘要: 大型语言模型（LLM）在训练和调整后会经历安全调整，但最近的工作表明，安全性可以通过越狱攻击绕过。虽然存在许多越狱和防御措施，但它们的跨语言概括仍然没有得到充分的研究。本文使用HarmBench和AdvBench上的六个LLM，首次对十种语言（跨越高、中和低资源语言）的越狱和防御进行了系统性的多语言评估。我们评估了两种越狱类型：基于逻辑表达的越狱和基于对抗提示的越狱。对于这两种类型，攻击成功率和防御稳健性因语言而异：高资源语言在标准查询下更安全，但更容易受到对抗查询的影响。简单的防御可能是有效的，但依赖于语言和模型。这些发现呼吁为LLM制定语言感知和跨语言安全基准。



## **12. Verifying LLM Inference to Prevent Model Weight Exfiltration**

CLARLLM推理以防止模型重量溢出 cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02620v1) [paper-pdf](http://arxiv.org/pdf/2511.02620v1)

**Authors**: Roy Rinberg, Adam Karvonen, Alex Hoover, Daniel Reuter, Keri Warr

**Abstract**: As large AI models become increasingly valuable assets, the risk of model weight exfiltration from inference servers grows accordingly. An attacker controlling an inference server may exfiltrate model weights by hiding them within ordinary model outputs, a strategy known as steganography. This work investigates how to verify model responses to defend against such attacks and, more broadly, to detect anomalous or buggy behavior during inference. We formalize model exfiltration as a security game, propose a verification framework that can provably mitigate steganographic exfiltration, and specify the trust assumptions associated with our scheme. To enable verification, we characterize valid sources of non-determinism in large language model inference and introduce two practical estimators for them. We evaluate our detection framework on several open-weight models ranging from 3B to 30B parameters. On MOE-Qwen-30B, our detector reduces exfiltratable information to <0.5% with false-positive rate of 0.01%, corresponding to a >200x slowdown for adversaries. Overall, this work further establishes a foundation for defending against model weight exfiltration and demonstrates that strong protection can be achieved with minimal additional cost to inference providers.

摘要: 随着大型人工智能模型成为越来越有价值的资产，模型权重从推理服务器泄露的风险也相应增加。控制推理服务器的攻击者可以通过将模型权重隐藏在普通模型输出中来溢出模型权重，这种策略称为隐写术。这项工作研究了如何验证模型响应以抵御此类攻击，以及更广泛地说，在推理过程中检测异常或有缺陷的行为。我们将模型溢出形式化为一个安全游戏，提出了一个可以证明减轻隐写溢出的验证框架，并指定与我们的方案相关的信任假设。为了实现验证，我们描述了大型语言模型推理中非决定性的有效来源，并为其引入了两个实用的估计器。我们在从3B到30 B参数的几个开放权重模型上评估了我们的检测框架。在MOE-Qwen-30 B上，我们的检测器将可渗透信息减少到<0.5%，假阳性率为0.01%，相当于对手的速度减慢> 200倍。总体而言，这项工作进一步奠定了防御模型权重溢出的基础，并证明可以以最小的额外成本来实现强大的保护。



## **13. The Dark Side of LLMs: Agent-based Attacks for Complete Computer Takeover**

LLM的阴暗面：基于代理的完全计算机接管攻击 cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2507.06850v5) [paper-pdf](http://arxiv.org/pdf/2507.06850v5)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables remarkable capabilities in natural language processing and generation. However, these systems introduce security vulnerabilities that extend beyond traditional content generation to system-level compromises. This paper presents a comprehensive evaluation of the LLMs security used as reasoning engines within autonomous agents, highlighting how they can be exploited as attack vectors capable of achieving computer takeovers. We focus on how different attack surfaces and trust boundaries can be leveraged to orchestrate such takeovers. We demonstrate that adversaries can effectively coerce popular LLMs into autonomously installing and executing malware on victim machines. Our evaluation of 18 state-of-the-art LLMs reveals an alarming scenario: 94.4% of models succumb to Direct Prompt Injection, and 83.3% are vulnerable to the more stealthy and evasive RAG Backdoor Attack. Notably, we tested trust boundaries within multi-agent systems, where LLM agents interact and influence each other, and we revealed that LLMs which successfully resist direct injection or RAG backdoor attacks will execute identical payloads when requested by peer agents. We found that 100.0% of tested LLMs can be compromised through Inter-Agent Trust Exploitation attacks, and that every model exhibits context-dependent security behaviors that create exploitable blind spots.

摘要: 大语言模型（LLM）代理和多代理系统的快速采用使自然语言处理和生成的能力显着。然而，这些系统引入了安全漏洞，这些安全漏洞超出了传统的内容生成，甚至会危及系统级安全。本文提出了一个全面的评估的LLM安全性作为推理引擎内的自主代理，突出它们如何可以被利用为攻击向量能够实现计算机接管。我们专注于如何利用不同的攻击面和信任边界来协调此类收购。我们证明，对手可以有效地强迫流行的LLM在受害者机器上自主安装和执行恶意软件。我们对18种最先进的LLM的评估揭示了一个令人震惊的情况：94.4%的模型屈服于直接提示注入，83.3%的模型容易受到更隐蔽和规避的RAG后门攻击。值得注意的是，我们测试了多代理系统中的信任边界，其中LLM代理相互交互和影响，我们揭示了成功抵抗直接注入或RAG后门攻击的LLM将在对等代理请求时执行相同的有效负载。我们发现，100.0%的测试LLM都可能通过代理间信任利用攻击而受到损害，并且每个模型都表现出依赖于上下文的安全行为，从而创建了可利用的盲点。



## **14. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

AutoAdv：大型语言模型多回合越狱的自动对抗预算 cs.CL

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02376v1) [paper-pdf](http://arxiv.org/pdf/2511.02376v1)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，其中对抗性提示会引发有害输出，但大多数评估都集中在单轮交互上，而现实世界的攻击则通过自适应多轮对话展开。我们介绍了AutoAdv，这是一个用于自动多回合越狱的免训练框架，在六个回合内对Llama-3.1-8B的攻击成功率高达95%，比单回合基线提高了24%。AutoAdv独特地结合了三种自适应机制：从成功的攻击中学习以增强未来提示的模式管理器、根据失败模式动态调整采样参数的温度管理器以及掩盖有害请求然后迭代细化它们的两阶段重写策略。对商业和开源模型（GPT-4 o-mini、Qwen 3 - 235 B、Mistral-7 B）的广泛评估揭示了当前安全机制中存在的持续漏洞，多回合攻击的表现始终优于单回合方法。这些发现表明，针对单轮交互优化的对齐策略无法在扩展对话中保持稳健性，凸显了对多轮感知防御的迫切需求。



## **15. An Automated Framework for Strategy Discovery, Retrieval, and Evolution in LLM Jailbreak Attacks**

LLM越狱攻击中策略发现、检索和进化的自动化框架 cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02356v1) [paper-pdf](http://arxiv.org/pdf/2511.02356v1)

**Authors**: Xu Liu, Yan Chen, Kan Ling, Yichi Zhu, Hengrun Zhang, Guisheng Fan, Huiqun Yu

**Abstract**: The widespread deployment of Large Language Models (LLMs) as public-facing web services and APIs has made their security a core concern for the web ecosystem. Jailbreak attacks, as one of the significant threats to LLMs, have recently attracted extensive research. In this paper, we reveal a jailbreak strategy which can effectively evade current defense strategies. It can extract valuable information from failed or partially successful attack attempts and contains self-evolution from attack interactions, resulting in sufficient strategy diversity and adaptability. Inspired by continuous learning and modular design principles, we propose ASTRA, a jailbreak framework that autonomously discovers, retrieves, and evolves attack strategies to achieve more efficient and adaptive attacks. To enable this autonomous evolution, we design a closed-loop "attack-evaluate-distill-reuse" core mechanism that not only generates attack prompts but also automatically distills and generalizes reusable attack strategies from every interaction. To systematically accumulate and apply this attack knowledge, we introduce a three-tier strategy library that categorizes strategies into Effective, Promising, and Ineffective based on their performance scores. The strategy library not only provides precise guidance for attack generation but also possesses exceptional extensibility and transferability. We conduct extensive experiments under a black-box setting, and the results show that ASTRA achieves an average Attack Success Rate (ASR) of 82.7%, significantly outperforming baselines.

摘要: 大型语言模型（LLM）作为面向公众的Web服务和API的广泛部署使其安全性成为Web生态系统的核心问题。越狱攻击作为LLM面临的主要威胁之一，近年来引起了广泛的研究。在本文中，我们揭示了一个越狱策略，可以有效地规避当前的防御策略。它可以从失败或部分成功的攻击尝试中提取有价值的信息，并包含来自攻击交互的自我进化，从而产生足够的策略多样性和适应性。受持续学习和模块化设计原则的启发，我们提出了ASTRA，这是一个越狱框架，可以自主发现、检索和进化攻击策略，以实现更高效和更适应性的攻击。为了实现这种自主进化，我们设计了一个闭环“攻击-评估-攻击-重用”核心机制，该机制不仅生成攻击提示，还自动从每次交互中提取和概括可重复使用的攻击策略。为了系统性地积累和应用这些攻击知识，我们引入了一个三层策略库，该库根据策略的性能得分将策略分为有效、有前途和无效。该策略库不仅为攻击生成提供精确的指导，而且具有出色的可扩展性和可移植性。我们在黑匣子环境下进行了广泛的实验，结果显示ASTRA的平均攻击成功率（ASB）为82.7%，显着优于基线。



## **16. Retrieval-Augmented Defense: Adaptive and Controllable Jailbreak Prevention for Large Language Models**

检索增强防御：大型语言模型的自适应且可控越狱预防 cs.CR

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2508.16406v2) [paper-pdf](http://arxiv.org/pdf/2508.16406v2)

**Authors**: Guangyu Yang, Jinghong Chen, Jingbiao Mei, Weizhe Lin, Bill Byrne

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreak attacks, which attempt to elicit harmful responses from LLMs. The evolving nature and diversity of these attacks pose many challenges for defense systems, including (1) adaptation to counter emerging attack strategies without costly retraining, and (2) control of the trade-off between safety and utility. To address these challenges, we propose Retrieval-Augmented Defense (RAD), a novel framework for jailbreak detection that incorporates a database of known attack examples into Retrieval-Augmented Generation, which is used to infer the underlying, malicious user query and jailbreak strategy used to attack the system. RAD enables training-free updates for newly discovered jailbreak strategies and provides a mechanism to balance safety and utility. Experiments on StrongREJECT show that RAD substantially reduces the effectiveness of strong jailbreak attacks such as PAP and PAIR while maintaining low rejection rates for benign queries. We propose a novel evaluation scheme and show that RAD achieves a robust safety-utility trade-off across a range of operating points in a controllable manner.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，这些攻击试图引发LLM的有害反应。这些攻击不断变化的性质和多样性给防御系统带来了许多挑战，包括（1）在无需昂贵的再培训的情况下适应应对新兴攻击策略，以及（2）控制安全性和实用性之间的权衡。为了应对这些挑战，我们提出了检索增强防御（RAD），这是一种新颖的越狱检测框架，将已知攻击示例的数据库整合到检索增强生成中，用于推断底层的恶意用户查询和用于攻击系统的越狱策略。RAD为新发现的越狱策略提供免训练更新，并提供平衡安全性和实用性的机制。StrongRESEARCH上的实验表明，RAD大大降低了PAP和PAIR等强越狱攻击的有效性，同时保持良性查询的低拒绝率。我们提出了一种新颖的评估方案，并表明RAD以可控的方式在一系列操作点上实现了稳健的安全-效用权衡。



## **17. Prompt Injection as an Emerging Threat: Evaluating the Resilience of Large Language Models**

提示注入作为一种新兴威胁：评估大型语言模型的弹性 cs.CR

10 pages, 6 figures

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01634v1) [paper-pdf](http://arxiv.org/pdf/2511.01634v1)

**Authors**: Daniyal Ganiuly, Assel Smaiyl

**Abstract**: Large Language Models (LLMs) are increasingly used in intelligent systems that perform reasoning, summarization, and code generation. Their ability to follow natural-language instructions, while powerful, also makes them vulnerable to a new class of attacks known as prompt injection. In these attacks, hidden or malicious instructions are inserted into user inputs or external content, causing the model to ignore its intended task or produce unsafe responses. This study proposes a unified framework for evaluating how resistant Large Language Models (LLMs) are to prompt injection attacks. The framework defines three complementary metrics such as the Resilience Degradation Index (RDI), Safety Compliance Coefficient (SCC), and Instructional Integrity Metric (IIM) to jointly measure robustness, safety, and semantic stability. We evaluated four instruction-tuned models (GPT-4, GPT-4o, LLaMA-3 8B Instruct, and Flan-T5-Large) on five common language tasks: question answering, summarization, translation, reasoning, and code generation. Results show that GPT-4 performs best overall, while open-weight models remain more vulnerable. The findings highlight that strong alignment and safety tuning are more important for resilience than model size alone. Results show that all models remain partially vulnerable, especially to indirect and direct-override attacks. GPT-4 achieved the best overall resilience (RDR = 9.8 %, SCR = 96.4 %), while open-source models exhibited higher performance degradation and lower safety scores. The findings demonstrate that alignment strength and safety tuning play a greater role in resilience than model size alone. The proposed framework offers a structured, reproducible approach for assessing model robustness and provides practical insights for improving LLM safety and reliability.

摘要: 大型语言模型（LLM）越来越多地用于执行推理、总结和代码生成的智能系统中。它们遵循自然语言指令的能力虽然强大，但也使它们容易受到称为提示注入的一类新型攻击。在这些攻击中，隐藏或恶意指令被插入到用户输入或外部内容中，导致模型忽略其预期任务或产生不安全的响应。这项研究提出了一个统一的框架来评估大型语言模型（LLM）对引发注入攻击的抵抗力。该框架定义了三个补充指标，例如弹性退化指数（RDI）、安全合规系数（SCC）和指令完整性指标（IIM），以联合衡量稳健性、安全性和语义稳定性。我们评估了四种经过翻译调整的模型（GPT-4、GPT-4 o、LLaMA-3 8B Direcct和Flan-T5-Large），用于五种常见语言任务：问题回答、总结、翻译、推理和代码生成。结果显示，GPT-4总体表现最好，而开重模型仍然更脆弱。研究结果强调，强对齐和安全调整对于弹性来说比模型尺寸更重要。结果表明，所有模型仍然部分容易受到攻击，尤其是受到间接和直接覆盖攻击的影响。GPT-4实现了最好的整体弹性（RDR = 9.8%，SCP = 96.4%），而开源模型表现出更高的性能退化和更低的安全评分。研究结果表明，对齐强度和安全调整比模型尺寸本身在弹性方面发挥更大的作用。提出的框架提供了一种结构化、可重复的方法来评估模型稳健性，并为提高LLM安全性和可靠性提供了实用见解。



## **18. Black-Box Membership Inference Attack for LVLMs via Prior Knowledge-Calibrated Memory Probing**

通过先验知识校准内存探测对LVLM进行黑匣子成员资格推断攻击 cs.CR

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01952v1) [paper-pdf](http://arxiv.org/pdf/2511.01952v1)

**Authors**: Jinhua Yin, Peiru Yang, Chen Yang, Huili Wang, Zhiyang Hu, Shangguang Wang, Yongfeng Huang, Tao Qi

**Abstract**: Large vision-language models (LVLMs) derive their capabilities from extensive training on vast corpora of visual and textual data. Empowered by large-scale parameters, these models often exhibit strong memorization of their training data, rendering them susceptible to membership inference attacks (MIAs). Existing MIA methods for LVLMs typically operate under white- or gray-box assumptions, by extracting likelihood-based features for the suspected data samples based on the target LVLMs. However, mainstream LVLMs generally only expose generated outputs while concealing internal computational features during inference, limiting the applicability of these methods. In this work, we propose the first black-box MIA framework for LVLMs, based on a prior knowledge-calibrated memory probing mechanism. The core idea is to assess the model memorization of the private semantic information embedded within the suspected image data, which is unlikely to be inferred from general world knowledge alone. We conducted extensive experiments across four LVLMs and three datasets. Empirical results demonstrate that our method effectively identifies training data of LVLMs in a purely black-box setting and even achieves performance comparable to gray-box and white-box methods. Further analysis reveals the robustness of our method against potential adversarial manipulations, and the effectiveness of the methodology designs. Our code and data are available at https://github.com/spmede/KCMP.

摘要: 大型视觉语言模型（LVLM）的能力源自对大量视觉和文本数据库的广泛培训。这些模型在大规模参数的支持下，通常表现出对其训练数据的强大记忆力，使其容易受到隶属推理攻击（MIA）。现有的LVLM MIA方法通常在白盒或灰盒假设下运行，通过基于目标LVLM为可疑数据样本提取基于似然的特征。然而，主流LVLM通常只暴露生成的输出，同时在推理过程中隐藏内部计算特征，从而限制了这些方法的适用性。在这项工作中，我们提出了第一个黑盒MIA框架LVLM，基于先验知识校准的内存探测机制。其核心思想是评估模型记忆的私人语义信息嵌入在可疑的图像数据，这是不可能推断出的一般世界知识。我们在四个LVLM和三个数据集上进行了广泛的实验。实验结果表明，我们的方法可以有效地识别LVLM的训练数据在一个纯粹的黑盒设置，甚至达到性能相媲美的灰盒和白盒方法。进一步的分析揭示了我们的方法对潜在的对抗性操纵的稳健性，以及方法论设计的有效性。我们的代码和数据可在https://github.com/spmede/KCMP上获取。



## **19. Align to Misalign: Automatic LLM Jailbreak with Meta-Optimized LLM Judges**

对齐与错位：通过元优化的LLM评委自动LLM越狱 cs.AI

under review, 28 pages

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01375v1) [paper-pdf](http://arxiv.org/pdf/2511.01375v1)

**Authors**: Hamin Koo, Minseon Kim, Jaehyung Kim

**Abstract**: Identifying the vulnerabilities of large language models (LLMs) is crucial for improving their safety by addressing inherent weaknesses. Jailbreaks, in which adversaries bypass safeguards with crafted input prompts, play a central role in red-teaming by probing LLMs to elicit unintended or unsafe behaviors. Recent optimization-based jailbreak approaches iteratively refine attack prompts by leveraging LLMs. However, they often rely heavily on either binary attack success rate (ASR) signals, which are sparse, or manually crafted scoring templates, which introduce human bias and uncertainty in the scoring outcomes. To address these limitations, we introduce AMIS (Align to MISalign), a meta-optimization framework that jointly evolves jailbreak prompts and scoring templates through a bi-level structure. In the inner loop, prompts are refined using fine-grained and dense feedback using a fixed scoring template. In the outer loop, the template is optimized using an ASR alignment score, gradually evolving to better reflect true attack outcomes across queries. This co-optimization process yields progressively stronger jailbreak prompts and more calibrated scoring signals. Evaluations on AdvBench and JBB-Behaviors demonstrate that AMIS achieves state-of-the-art performance, including 88.0% ASR on Claude-3.5-Haiku and 100.0% ASR on Claude-4-Sonnet, outperforming existing baselines by substantial margins.

摘要: 识别大型语言模型（LLM）的漏洞对于通过解决固有弱点来提高其安全性至关重要。越狱是指对手通过精心设计的输入提示绕过保障措施，通过探测LLM以引发意外或不安全的行为，在红色团队中发挥着核心作用。最近的基于优化的越狱方法通过利用LLM迭代地完善攻击提示。然而，他们通常严重依赖稀疏的二元攻击成功率（ASB）信号或手动制作的评分模板，这在评分结果中引入了人为偏见和不确定性。为了解决这些限制，我们引入了AMIS（Align to MISign），这是一个元优化框架，通过两级结构联合开发越狱提示和评分模板。在内循环中，使用固定评分模板使用细粒度且密集的反馈来细化提示。在外循环中，使用ASB对齐分数对模板进行优化，逐渐发展以更好地反映跨查询的真实攻击结果。这个协同优化过程会产生越来越强的越狱提示和更校准的评分信号。对AdvBench和JBB-Behavior的评估表明，AMIS实现了最先进的性能，包括Claude-3.5-Haiku的ASB为88.0%，Claude-4-Sonnet的ASB为100.0%，大幅优于现有基线。



## **20. Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems**

拯救未中毒者：有效防御RAG系统知识腐败攻击 cs.CR

15 pages, 7 figures, 10 tables. To appear in the Proceedings of the  2025 Annual Computer Security Applications Conference (ACSAC)

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01268v1) [paper-pdf](http://arxiv.org/pdf/2511.01268v1)

**Authors**: Minseok Kim, Hankook Lee, Hyungjoon Koo

**Abstract**: Large language models (LLMs) are reshaping numerous facets of our daily lives, leading widespread adoption as web-based services. Despite their versatility, LLMs face notable challenges, such as generating hallucinated content and lacking access to up-to-date information. Lately, to address such limitations, Retrieval-Augmented Generation (RAG) has emerged as a promising direction by generating responses grounded in external knowledge sources. A typical RAG system consists of i) a retriever that probes a group of relevant passages from a knowledge base and ii) a generator that formulates a response based on the retrieved content. However, as with other AI systems, recent studies demonstrate the vulnerability of RAG, such as knowledge corruption attacks by injecting misleading information. In response, several defense strategies have been proposed, including having LLMs inspect the retrieved passages individually or fine-tuning robust retrievers. While effective, such approaches often come with substantial computational costs.   In this work, we introduce RAGDefender, a resource-efficient defense mechanism against knowledge corruption (i.e., by data poisoning) attacks in practical RAG deployments. RAGDefender operates during the post-retrieval phase, leveraging lightweight machine learning techniques to detect and filter out adversarial content without requiring additional model training or inference. Our empirical evaluations show that RAGDefender consistently outperforms existing state-of-the-art defenses across multiple models and adversarial scenarios: e.g., RAGDefender reduces the attack success rate (ASR) against the Gemini model from 0.89 to as low as 0.02, compared to 0.69 for RobustRAG and 0.24 for Discern-and-Answer when adversarial passages outnumber legitimate ones by a factor of four (4x).

摘要: 大型语言模型（LLM）正在重塑我们日常生活的多个方面，并作为基于网络的服务广泛采用。尽管LLM具有多功能性，但仍面临着显着的挑战，例如生成幻觉内容和缺乏对最新信息的访问。最近，为了解决这些限制，检索增强一代（RAG）通过生成基于外部知识来源的响应而成为一个有前途的方向。典型的RAG系统由i）从知识库中探测一组相关段落的检索器和ii）根据检索到的内容制定响应的生成器组成。然而，与其他人工智能系统一样，最近的研究表明了RAG的脆弱性，例如通过注入误导性信息进行知识腐败攻击。作为回应，人们提出了几种防御策略，包括让LLM单独检查检索到的段落或微调稳健的检索器。虽然有效，但此类方法通常具有巨大的计算成本。   在这项工作中，我们引入了RAGDefender，这是一种针对知识腐败的资源高效防御机制（即通过数据中毒）实际RAG部署中的攻击。RAGDefender在检索后阶段运行，利用轻量级机器学习技术来检测和过滤对抗性内容，而无需额外的模型训练或推理。我们的实证评估表明，RAGDefender在多种模型和对抗场景中始终优于现有的最先进防御：例如，RAGDefender将针对Gemini模型的攻击成功率（ASB）从0.89降低到低至0.02，而当对抗性段落数量超过合法段落四倍（4x）时，RobustRAG的攻击成功率为0.69，辨别并回答的攻击成功率为0.24。



## **21. MistralBSM: Leveraging Mistral-7B for Vehicular Networks Misbehavior Detection**

MistralBSM：利用Mistral-7 B进行车辆网络不当行为检测 cs.LG

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2407.18462v2) [paper-pdf](http://arxiv.org/pdf/2407.18462v2)

**Authors**: Wissal Hamhoum, Soumaya Cherkaoui

**Abstract**: Malicious attacks on vehicular networks pose a serious threat to road safety as well as communication reliability. A major source of these threats stems from misbehaving vehicles within the network. To address this challenge, we propose a Large Language Model (LLM)-empowered Misbehavior Detection System (MDS) within an edge-cloud detection framework. Specifically, we fine-tune Mistral-7B, a compact and high-performing LLM, to detect misbehavior based on Basic Safety Messages (BSM) sequences as the edge component for real-time detection, while a larger LLM deployed in the cloud validates and reinforces the edge model's detection through a more comprehensive analysis. By updating only 0.012% of the model parameters, our model, which we named MistralBSM, achieves 98% accuracy in binary classification and 96% in multiclass classification on a selected set of attacks from VeReMi dataset, outperforming LLAMA2-7B and RoBERTa. Our results validate the potential of LLMs in MDS, showing a significant promise in strengthening vehicular network security to better ensure the safety of road users.

摘要: 对车辆网络的恶意攻击对道路安全和通信可靠性构成严重威胁。这些威胁的主要来源来自网络内行为不当的车辆。为了应对这一挑战，我们在边缘云检测框架内提出了一个支持大语言模型（LLM）的不当行为检测系统（SCS）。具体来说，我们对Mistral-7 B（一种紧凑且高性能的LLM）进行了微调，以基于基本安全消息（BSM）序列检测不当行为，作为实时检测的边缘组件，而部署在云中的更大LLM则通过更全面的分析验证和加强边缘模型的检测。通过仅更新0.012%的模型参数，我们的模型（我们命名为MistralBSM）对来自VeReMi数据集的一组选定的攻击在二进制分类中实现了98%的准确率和96%的多类分类准确率，优于LLAMA 2 - 7 B和RoBERTa。我们的结果验证了LLM在CMS中的潜力，显示出在加强车辆网络安全以更好地确保道路使用者安全方面的巨大前景。



## **22. Exploring the limits of strong membership inference attacks on large language models**

探索大型语言模型强隶属度推理攻击的局限性 cs.CR

NeurIPS 2025

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2505.18773v2) [paper-pdf](http://arxiv.org/pdf/2505.18773v2)

**Authors**: Jamie Hayes, Ilia Shumailov, Christopher A. Choquette-Choo, Matthew Jagielski, George Kaissis, Milad Nasr, Sahra Ghalebikesabi, Meenatchi Sundaram Mutu Selva Annamalai, Niloofar Mireshghallah, Igor Shilov, Matthieu Meeus, Yves-Alexandre de Montjoye, Katherine Lee, Franziska Boenisch, Adam Dziedzic, A. Feder Cooper

**Abstract**: State-of-the-art membership inference attacks (MIAs) typically require training many reference models, making it difficult to scale these attacks to large pre-trained language models (LLMs). As a result, prior research has either relied on weaker attacks that avoid training references (e.g., fine-tuning attacks), or on stronger attacks applied to small models and datasets. However, weaker attacks have been shown to be brittle and insights from strong attacks in simplified settings do not translate to today's LLMs. These challenges prompt an important question: are the limitations observed in prior work due to attack design choices, or are MIAs fundamentally ineffective on LLMs? We address this question by scaling LiRA--one of the strongest MIAs--to GPT-2 architectures ranging from 10M to 1B parameters, training references on over 20B tokens from the C4 dataset. Our results advance the understanding of MIAs on LLMs in four key ways. While (1) strong MIAs can succeed on pre-trained LLMs, (2) their effectiveness, remains limited (e.g., AUC<0.7) in practical settings. (3) Even when strong MIAs achieve better-than-random AUC, aggregate metrics can conceal substantial per-sample MIA decision instability: due to training randomness, many decisions are so unstable that they are statistically indistinguishable from a coin flip. Finally, (4) the relationship between MIA success and related LLM privacy metrics is not as straightforward as prior work has suggested.

摘要: 最先进的成员资格推理攻击（MIA）通常需要训练许多参考模型，因此很难将这些攻击扩展到大型预训练语言模型（LLM）。因此，之前的研究要么依赖于避免训练参考的较弱攻击（例如，微调攻击），或者应用于小型模型和数据集的更强攻击。然而，较弱的攻击已被证明是脆弱的，并且在简化环境中来自强攻击的见解无法转化为当今的LLM。这些挑战提出了一个重要的问题：在之前的工作中观察到的限制是由于攻击设计选择造成的，还是MIA从根本上对LLM无效？我们通过将LiRA（最强大的MIA之一）扩展到GPT-2架构来解决这个问题，参数范围从10 M到1B，并在来自C4数据集中的超过20 B个令牌上训练引用。我们的结果通过四种关键方式促进了对LLM上MIA的理解。虽然（1）强大的MIA可以在预先培训的LLM上取得成功，但（2）它们的有效性仍然有限（例如，在实际环境中，AUC < 0.7）。(3)即使当强MIA实现优于随机的UC时，聚合指标也可能掩盖每个样本MIA决策的实质性不稳定性：由于训练随机性，许多决策非常不稳定，以至于在统计上与抛硬币没有区别。最后，（4）MIA成功与相关LLM隐私指标之间的关系并不像之前的工作表明的那么简单。



## **23. SafeDialBench: A Fine-Grained Safety Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks**

SafeDialBench：针对具有多样越狱攻击的多回合对话中大型语言模型的细粒度安全基准 cs.CL

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2502.11090v3) [paper-pdf](http://arxiv.org/pdf/2502.11090v3)

**Authors**: Hongye Cao, Yanming Wang, Sijia Jing, Ziyue Peng, Zhixin Bai, Zhe Cao, Meng Fang, Fan Feng, Boyan Wang, Jiaheng Liu, Tianpei Yang, Jing Huo, Yang Gao, Fanyu Meng, Xi Yang, Chao Deng, Junlan Feng

**Abstract**: With the rapid advancement of Large Language Models (LLMs), the safety of LLMs has been a critical concern requiring precise assessment. Current benchmarks primarily concentrate on single-turn dialogues or a single jailbreak attack method to assess the safety. Additionally, these benchmarks have not taken into account the LLM's capability of identifying and handling unsafe information in detail. To address these issues, we propose a fine-grained benchmark SafeDialBench for evaluating the safety of LLMs across various jailbreak attacks in multi-turn dialogues. Specifically, we design a two-tier hierarchical safety taxonomy that considers 6 safety dimensions and generates more than 4000 multi-turn dialogues in both Chinese and English under 22 dialogue scenarios. We employ 7 jailbreak attack strategies, such as reference attack and purpose reverse, to enhance the dataset quality for dialogue generation. Notably, we construct an innovative assessment framework of LLMs, measuring capabilities in detecting, and handling unsafe information and maintaining consistency when facing jailbreak attacks. Experimental results across 17 LLMs reveal that Yi-34B-Chat and GLM4-9B-Chat demonstrate superior safety performance, while Llama3.1-8B-Instruct and o3-mini exhibit safety vulnerabilities.

摘要: 随着大型语言模型（LLM）的迅速发展，LLM的安全性已成为需要精确评估的关键问题。目前的基准主要集中在单回合对话或单次越狱攻击方法上来评估安全性。此外，这些基准没有考虑LLM详细识别和处理不安全信息的能力。为了解决这些问题，我们提出了一个细粒度基准SafeDialBench，用于评估LLM在多回合对话中在各种越狱攻击中的安全性。具体来说，我们设计了一个两层分层安全分类法，考虑6个安全维度，并在22个对话场景下生成4000多个中英多回合对话。我们采用引用攻击和目的反向等7种越狱攻击策略来提高对话生成的数据集质量。值得注意的是，我们构建了一个创新的LLM评估框架，衡量检测和处理不安全信息以及在面临越狱攻击时保持一致性的能力。17个LLM的实验结果显示，Yi-34 B-Chat和GLM 4 - 9 B-Chat表现出出色的安全性能，而Llama3.1- 8B-Direct和o3-mini则表现出安全漏洞。



## **24. Enhancing Adversarial Transferability in Visual-Language Pre-training Models via Local Shuffle and Sample-based Attack**

通过本地洗牌和基于样本的攻击增强视觉语言预训练模型中的对抗可移植性 cs.CV

Accepted by NAACL2025 findings

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2511.00831v1) [paper-pdf](http://arxiv.org/pdf/2511.00831v1)

**Authors**: Xin Liu, Aoyang Zhou, Aoyang Zhou

**Abstract**: Visual-Language Pre-training (VLP) models have achieved significant performance across various downstream tasks. However, they remain vulnerable to adversarial examples. While prior efforts focus on improving the adversarial transferability of multimodal adversarial examples through cross-modal interactions, these approaches suffer from overfitting issues, due to a lack of input diversity by relying excessively on information from adversarial examples in one modality when crafting attacks in another. To address this issue, we draw inspiration from strategies in some adversarial training methods and propose a novel attack called Local Shuffle and Sample-based Attack (LSSA). LSSA randomly shuffles one of the local image blocks, thus expanding the original image-text pairs, generating adversarial images, and sampling around them. Then, it utilizes both the original and sampled images to generate the adversarial texts. Extensive experiments on multiple models and datasets demonstrate that LSSA significantly enhances the transferability of multimodal adversarial examples across diverse VLP models and downstream tasks. Moreover, LSSA outperforms other advanced attacks on Large Vision-Language Models.

摘要: 视觉语言预训练（VLP）模型在各种下游任务中取得了显着的性能。然而，他们仍然容易受到敌对例子的影响。虽然之前的工作重点是通过跨模式交互来提高多模式对抗示例的对抗可转移性，但这些方法存在过度匹配问题，因为在设计另一种模式时过度依赖来自对抗示例的信息，从而缺乏输入多样性。为了解决这个问题，我们从一些对抗训练方法中的策略中汲取灵感，并提出了一种名为本地洗牌和基于样本的攻击（LSCA）的新型攻击。LSSA随机洗牌其中一个本地图像块，从而扩展原始图像-文本对，生成对抗图像，并对其进行抽样。然后，它利用原始图像和采样图像来生成对抗文本。对多个模型和数据集的广泛实验表明，LSA显着增强了多模式对抗示例在不同VLP模型和下游任务之间的可移植性。此外，LSCA的性能优于对大型视觉语言模型的其他高级攻击。



## **25. Adversarial Déjà Vu: Jailbreak Dictionary Learning for Stronger Generalization to Unseen Attacks**

对抗性Déjà Vu：越狱词典学习以更强有力地概括隐形攻击 cs.LG

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2510.21910v2) [paper-pdf](http://arxiv.org/pdf/2510.21910v2)

**Authors**: Mahavir Dabas, Tran Huynh, Nikhil Reddy Billa, Jiachen T. Wang, Peng Gao, Charith Peris, Yao Ma, Rahul Gupta, Ming Jin, Prateek Mittal, Ruoxi Jia

**Abstract**: Large language models remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Defending against novel jailbreaks represents a critical challenge in AI safety. Adversarial training -- designed to make models robust against worst-case perturbations -- has been the dominant paradigm for adversarial robustness. However, due to optimization challenges and difficulties in defining realistic threat models, adversarial training methods often fail on newly developed jailbreaks in practice. This paper proposes a new paradigm for improving robustness against unseen jailbreaks, centered on the Adversarial D\'ej\`a Vu hypothesis: novel jailbreaks are not fundamentally new, but largely recombinations of adversarial skills from previous attacks. We study this hypothesis through a large-scale analysis of 32 attack papers published over two years. Using an automated pipeline, we extract and compress adversarial skills into a sparse dictionary of primitives, with LLMs generating human-readable descriptions. Our analysis reveals that unseen attacks can be effectively explained as sparse compositions of earlier skills, with explanatory power increasing monotonically as skill coverage grows. Guided by this insight, we introduce Adversarial Skill Compositional Training (ASCoT), which trains on diverse compositions of skill primitives rather than isolated attack instances. ASCoT substantially improves robustness to unseen attacks, including multi-turn jailbreaks, while maintaining low over-refusal rates. We also demonstrate that expanding adversarial skill coverage, not just data scale, is key to defending against novel attacks. \textcolor{red}{\textbf{Warning: This paper contains content that may be harmful or offensive in nature.

摘要: 大型语言模型仍然容易受到越狱攻击，这些攻击绕过安全护栏，引发有害输出。防御新型越狱是人工智能安全的一个关键挑战。对抗性训练--旨在使模型在最坏情况下保持稳健性--一直是对抗性稳健性的主要范式。然而，由于优化挑战和定义现实威胁模型的困难，对抗性训练方法在实践中常常在新开发的越狱中失败。本文以对抗D ' ej ' a Vu假设为中心，提出了一种用于提高针对未见越狱的鲁棒性的新范式：新型越狱从根本上来说并不是新的，而是之前攻击中对抗技能的重新组合。我们通过对两年内发表的32篇攻击论文的大规模分析来研究这一假设。使用自动化管道，我们将对抗技能提取并压缩到稀疏的基元字典中，由LLM生成人类可读的描述。我们的分析表明，不可见的攻击可以有效地解释为早期技能的稀疏组成，并且随着技能覆盖范围的增加，解释能力单调增加。在这一见解的指导下，我们引入了对抗性技能合成训练（ASCoT），该训练基于技能基元的不同合成，而不是孤立的攻击实例。ASCoT大幅提高了对不可见攻击（包括多回合越狱）的稳健性，同时保持较低的过度拒绝率。我们还证明，扩大对抗技能覆盖范围，而不仅仅是数据规模，是防御新型攻击的关键。\textColor{red}{\textBF{警告：本文包含可能有害或冒犯性的内容。



## **26. ShadowLogic: Backdoors in Any Whitebox LLM**

ShadowLogic：任何白盒LLM中的后门 cs.CR

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00664v1) [paper-pdf](http://arxiv.org/pdf/2511.00664v1)

**Authors**: Kasimir Schulz, Amelia Kawasaki, Leo Ring

**Abstract**: Large language models (LLMs) are widely deployed across various applications, often with safeguards to prevent the generation of harmful or restricted content. However, these safeguards can be covertly bypassed through adversarial modifications to the computational graph of a model. This work highlights a critical security vulnerability in computational graph-based LLM formats, demonstrating that widely used deployment pipelines may be susceptible to obscured backdoors. We introduce ShadowLogic, a method for creating a backdoor in a white-box LLM by injecting an uncensoring vector into its computational graph representation. We set a trigger phrase that, when added to the beginning of a prompt into the LLM, applies the uncensoring vector and removes the content generation safeguards in the model. We embed trigger logic directly into the computational graph which detects the trigger phrase in a prompt. To evade detection of our backdoor, we obfuscate this logic within the graph structure, making it similar to standard model functions. Our method requires minimal alterations to model parameters, making backdoored models appear benign while retaining the ability to generate uncensored responses when activated. We successfully implement ShadowLogic in Phi-3 and Llama 3.2, using ONNX for manipulating computational graphs. Implanting the uncensoring vector achieved a >60% attack success rate for further malicious queries.

摘要: 大型语言模型（LLM）广泛部署在各种应用程序中，通常具有防止产生有害或受限制内容的保护措施。然而，这些保障措施可以通过对模型计算图的对抗性修改而被秘密绕过。这项工作强调了基于计算图的LLM格式中的一个关键安全漏洞，证明广泛使用的部署管道可能容易受到隐藏的后门的影响。我们引入ShadowLogic，这是一种通过将未经审查的载体注入到其计算图表示中来在白盒LLM中创建后门的方法。我们设置了一个触发短语，当将其添加到LLM中的提示符的开头时，将应用未审查向量并删除模型中的内容生成保护措施。我们将触发逻辑直接嵌入到计算图中，该计算图检测提示中的触发短语。为了避免后门被检测到，我们在图结构中混淆了这个逻辑，使其类似于标准模型函数。我们的方法需要对模型参数进行最小的更改，使后门模型看起来是良性的，同时保留在激活时生成未经审查的响应的能力。我们在Phi-3和Llama 3.2中成功实现了ShadowLogic，使用ONNX来操纵计算图。植入未经审查的载体对于进一步的恶意查询的攻击成功率超过60%。



## **27. SLIP: Securing LLMs IP Using Weights Decomposition**

SIP：使用权重分解保护LLM IP cs.CR

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2407.10886v3) [paper-pdf](http://arxiv.org/pdf/2407.10886v3)

**Authors**: Yehonathan Refael, Adam Hakim, Lev Greenberg, Satya Lokam, Tal Aviv, Ben Fishman, Shachar Seidman, Racchit Jain, Jay Tenenbaum

**Abstract**: Large language models (LLMs) have recently seen widespread adoption in both academia and industry. As these models grow, they become valuable intellectual property (IP), reflecting substantial investments by their owners. The high cost of cloud-based deployment has spurred interest in running models on edge devices, but this risks exposing parameters to theft and unauthorized use. Existing approaches to protect model IP on the edge trade off practicality, accuracy, or deployment requirements. We introduce SLIP, a hybrid inference algorithm designed to protect edge-deployed models from theft. SLIP is, to our knowledge, the first hybrid protocol that is both practical for real-world applications and provably secure, while incurring zero accuracy degradation and minimal latency overhead. It partitions the model across two computing resources: one secure but expensive, and one cost-effective but vulnerable. Using matrix decomposition, the secure resource retains the most sensitive portion of the model's IP while performing only a small fraction of the computation; the vulnerable resource executes the remainder. The protocol includes security guarantees that prevent attackers from using the partition to infer the protected information. Finally, we present experimental results that demonstrate the robustness and effectiveness of our method, positioning it as a compelling solution for protecting LLMs.

摘要: 大型语言模型（LLM）最近在学术界和工业界得到了广泛采用。随着这些模型的发展，它们成为宝贵的知识产权（IP），反映了其所有者的大量投资。基于云的部署的高成本激发了人们对在边缘设备上运行模型的兴趣，但这可能会导致参数被盗和未经授权使用。保护边缘模型IP的现有方法权衡了实用性、准确性或部署要求。我们引入了SIIP，这是一种混合推理算法，旨在保护边缘部署的模型免受盗窃。据我们所知，SlIP是第一个既适用于现实世界应用程序又可证明安全的混合协议，同时会导致零准确度下降和最小的延迟负担。它将模型划分为两种计算资源：一种安全但昂贵，另一种经济实惠但容易受到攻击。使用矩阵分解，安全资源保留模型IP中最敏感的部分，同时仅执行一小部分计算;易受攻击的资源执行其余部分。该协议包括安全保证，防止攻击者使用分区来推断受保护的信息。最后，我们提供了实验结果，证明了我们方法的鲁棒性和有效性，将其定位为保护LLM的令人信服的解决方案。



## **28. What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks**

Prettts越狱LLMS有哪些功能？调查攻击背后的机制 cs.CR

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2411.03343v3) [paper-pdf](http://arxiv.org/pdf/2411.03343v3)

**Authors**: Nathalie Kirch, Constantin Weisser, Severin Field, Helen Yannakoudakis, Stephen Casper

**Abstract**: Jailbreaks have been a central focus of research regarding the safety and reliability of large language models (LLMs), yet the mechanisms underlying these attacks remain poorly understood. While previous studies have predominantly relied on linear methods to detect jailbreak attempts and model refusals, we take a different approach by examining both linear and non-linear features in prompts that lead to successful jailbreaks. First, we introduce a novel dataset comprising 10,800 jailbreak attempts spanning 35 diverse attack methods. Leveraging this dataset, we train linear and non-linear probes on hidden states of open-weight LLMs to predict jailbreak success. Probes achieve strong in-distribution accuracy but transfer is attack-family-specific, revealing that different jailbreaks are supported by distinct internal mechanisms rather than a single universal direction. To establish causal relevance, we construct probe-guided latent interventions that systematically shift compliance in the predicted direction. Interventions derived from non-linear probes produce larger and more reliable effects than those from linear probes, indicating that features linked to jailbreak success are encoded non-linearly in prompt representations. Overall, the results surface heterogeneous, non-linear structure in jailbreak mechanisms and provide a prompt-side methodology for recovering and testing the features that drive jailbreak outcomes.

摘要: 越狱一直是大型语言模型（LLM）安全性和可靠性研究的中心焦点，但人们对这些攻击的潜在机制仍然知之甚少。虽然之前的研究主要依赖线性方法来检测越狱尝试和模型拒绝，但我们采取了不同的方法，通过检查导致成功越狱的提示中的线性和非线性特征。首先，我们引入了一个新颖的数据集，其中包含10，800次越狱尝试，涵盖35种不同的攻击方法。利用该数据集，我们对开重轻量级LLM的隐藏状态训练线性和非线性探针，以预测越狱成功。探针实现了很强的内部分布准确性，但转移是针对攻击家庭的，这表明不同的越狱是由不同的内部机制而不是单一的普遍方向支持的。为了建立因果相关性，我们构建了探针引导的潜在干预措施，系统性地将依从性转向预测方向。来自非线性探针的干预措施比来自线性探针的干预措施产生更大、更可靠的效果，这表明与越狱成功相关的特征在提示表示中被非线性编码。总体而言，这些结果揭示了越狱机制中的异类、非线性结构，并提供了一种预算侧方法论来恢复和测试驱动越狱结果的功能。



## **29. Friend or Foe: How LLMs' Safety Mind Gets Fooled by Intent Shift Attack**

朋友还是敌人：法学硕士的安全思想如何被意图转变攻击所愚弄 cs.CL

Preprint, 14 pages, 5 figures, 7 tables

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00556v1) [paper-pdf](http://arxiv.org/pdf/2511.00556v1)

**Authors**: Peng Ding, Jun Kuang, Wen Sun, Zongyu Wang, Xuezhi Cao, Xunliang Cai, Jiajun Chen, Shujian Huang

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreaking attacks despite their impressive capabilities. Investigating these weaknesses is crucial for robust safety mechanisms. Existing attacks primarily distract LLMs by introducing additional context or adversarial tokens, leaving the core harmful intent unchanged. In this paper, we introduce ISA (Intent Shift Attack), which obfuscates LLMs about the intent of the attacks. More specifically, we establish a taxonomy of intent transformations and leverage them to generate attacks that may be misperceived by LLMs as benign requests for information. Unlike prior methods relying on complex tokens or lengthy context, our approach only needs minimal edits to the original request, and yields natural, human-readable, and seemingly harmless prompts. Extensive experiments on both open-source and commercial LLMs show that ISA achieves over 70% improvement in attack success rate compared to direct harmful prompts. More critically, fine-tuning models on only benign data reformulated with ISA templates elevates success rates to nearly 100%. For defense, we evaluate existing methods and demonstrate their inadequacy against ISA, while exploring both training-free and training-based mitigation strategies. Our findings reveal fundamental challenges in intent inference for LLMs safety and underscore the need for more effective defenses. Our code and datasets are available at https://github.com/NJUNLP/ISA.

摘要: 大型语言模型（LLM）尽管具有令人印象深刻的功能，但仍然容易受到越狱攻击。调查这些弱点对于强大的安全机制至关重要。现有的攻击主要通过引入额外的上下文或对抗性代币来分散LLM的注意力，从而保持核心有害意图不变。在本文中，我们介绍了ISA（意图转移攻击），它使LLM混淆了攻击意图。更具体地说，我们建立了意图转换的分类法，并利用它们来生成可能被LLM误认为是良性信息请求的攻击。与依赖复杂令牌或冗长上下文的现有方法不同，我们的方法只需要对原始请求进行最少的编辑，并产生自然、人类可读且看似无害的提示。对开源和商业LLM的广泛实验表明，与直接有害提示相比，ISA的攻击成功率提高了70%以上。更关键的是，仅对使用ISA模板重新制定的良性数据进行微调模型，可以将成功率提高到近100%。对于防御，我们评估现有方法并证明其在ISA中的不足之处，同时探索免训练和基于训练的缓解策略。我们的研究结果揭示了LLM安全性意图推理的根本挑战，并强调了更有效防御的必要性。我们的代码和数据集可在https://github.com/NJUNLP/ISA上获取。



## **30. Reimagining Safety Alignment with An Image**

与形象重新构想安全性 cs.AI

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00509v1) [paper-pdf](http://arxiv.org/pdf/2511.00509v1)

**Authors**: Yifan Xia, Guorui Chen, Wenqian Yu, Zhijiang Li, Philip Torr, Jindong Gu

**Abstract**: Large language models (LLMs) excel in diverse applications but face dual challenges: generating harmful content under jailbreak attacks and over-refusal of benign queries due to rigid safety mechanisms. These issues are further complicated by the need to accommodate different value systems and precisely align with given safety preferences. Moreover, traditional methods like SFT and RLHF lack this capability due to their costly parameter tuning requirements and inability to support multiple value systems within a single model. These problems are more obvious in multimodal large language models (MLLMs), especially in terms of heightened over-refusal in cross-modal tasks and new security risks arising from expanded attack surfaces. We propose Magic Image, an optimization-driven visual prompt framework that enhances security while reducing over-refusal. By optimizing image prompts using harmful/benign samples, our method enables a single model to adapt to different value systems and better align with given safety preferences without parameter updates. Experiments demonstrate improved safety-effectiveness balance across diverse datasets while preserving model performance, offering a practical solution for deployable MLLM safety alignment.

摘要: 大型语言模型（LLM）在不同的应用程序中表现出色，但面临双重挑战：在越狱攻击下生成有害内容以及由于严格的安全机制而过度拒绝良性查询。由于需要适应不同的价值体系并与给定的安全偏好精确保持一致，这些问题变得更加复杂。此外，传统的方法，如SFT和RLHF缺乏这种能力，由于其昂贵的参数调整要求和无法支持多个价值系统在一个单一的模型。这些问题在多模态大型语言模型（MLLM）中更为明显，特别是在跨模态任务中的过度拒绝和扩展的攻击面带来的新的安全风险方面。我们提出了魔术图像，优化驱动的视觉提示框架，提高了安全性，同时减少过度拒绝。通过使用有害/良性样本优化图像提示，我们的方法使单个模型能够适应不同的价值体系，并在不更新参数的情况下更好地与给定的安全偏好保持一致。实验证明，不同数据集的安全性-有效性平衡得到了改善，同时保持了模型性能，为可部署的MLLM安全对齐提供了实用的解决方案。



## **31. Proactive DDoS Detection and Mitigation in Decentralized Software-Defined Networking via Port-Level Monitoring and Zero-Training Large Language Models**

通过端口级监控和零训练大型语言模型在分散式软件定义网络中主动检测和缓解 cs.CR

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00460v1) [paper-pdf](http://arxiv.org/pdf/2511.00460v1)

**Authors**: Mohammed N. Swileh, Shengli Zhang

**Abstract**: Centralized Software-Defined Networking (cSDN) offers flexible and programmable control of networks but suffers from scalability and reliability issues due to its reliance on centralized controllers. Decentralized SDN (dSDN) alleviates these concerns by distributing control across multiple local controllers, yet this architecture remains highly vulnerable to Distributed Denial-of-Service (DDoS) attacks. In this paper, we propose a novel detection and mitigation framework tailored for dSDN environments. The framework leverages lightweight port-level statistics combined with prompt engineering and in-context learning, enabling the DeepSeek-v3 Large Language Model (LLM) to classify traffic as benign or malicious without requiring fine-tuning or retraining. Once an anomaly is detected, mitigation is enforced directly at the attacker's port, ensuring that malicious traffic is blocked at their origin while normal traffic remains unaffected. An automatic recovery mechanism restores normal operation after the attack inactivity, ensuring both security and availability. Experimental evaluation under diverse DDoS attack scenarios demonstrates that the proposed approach achieves near-perfect detection, with 99.99% accuracy, 99.97% precision, 100% recall, 99.98% F1-score, and an AUC of 1.0. These results highlight the effectiveness of combining distributed monitoring with zero-training LLM inference, providing a proactive and scalable defense mechanism for securing dSDN infrastructures against DDoS threats.

摘要: 集中式软件定义网络（cdn）提供灵活且可编程的网络控制，但由于依赖集中式控制器，存在可扩展性和可靠性问题。去中心化的dn（ddn）通过将控制权分布在多个本地控制器上来缓解这些担忧，但该架构仍然极易受到分布式拒绝服务（DDOS）攻击的影响。在本文中，我们提出了一种专为ddn环境量身定制的新型检测和缓解框架。该框架利用轻量级的端口级统计数据，结合即时工程和上下文学习，使DeepSeek-v3大型语言模型（LLM）能够将流量分类为良性或恶意，而无需微调或重新训练。一旦检测到异常，就会直接在攻击者的端口上实施缓解措施，确保恶意流量在其来源处被拦截，而正常流量不受影响。自动恢复机制在攻击不活动后恢复正常操作，确保安全性和可用性。在不同的DDOS攻击场景下的实验评估表明，所提出的方法实现了近乎完美的检测，准确率为99.99%，准确率为99.97%，召回率为100%，F1评分为99.98%，AUC1.0。这些结果凸显了将分布式监控与零训练LLM推断相结合的有效性，从而提供了一种主动且可扩展的防御机制，以保护ddn基础设施免受DDOS威胁。



## **32. DRIP: Defending Prompt Injection via De-instruction Training and Residual Fusion Model Architecture**

DRIP：通过去指令训练和剩余融合模型架构来防御即时注入 cs.CR

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00447v1) [paper-pdf](http://arxiv.org/pdf/2511.00447v1)

**Authors**: Ruofan Liu, Yun Lin, Jin Song Dong

**Abstract**: Large language models (LLMs) have demonstrated impressive instruction-following capabilities. However, these capabilities also expose models to prompt injection attacks, where maliciously crafted inputs overwrite or distract from the intended instructions. A core vulnerability lies in the model's lack of semantic role understanding: it cannot distinguish directive intent from descriptive content, leading it to execute instruction-like phrases embedded in data.   We propose DRIP, a training-time defense grounded in a semantic modeling perspective, which enforces robust separation between instruction and data semantics without sacrificing utility. DRIP introduces two lightweight yet complementary mechanisms: (1) a token-wise de-instruction shift that performs semantic disentanglement, weakening directive semantics in data tokens while preserving content meaning; and (2) a residual fusion pathway that provides a persistent semantic anchor, reinforcing the influence of the true top-level instruction during generation. Experimental results on LLaMA-8B and Mistral-7B across three prompt injection benchmarks (SEP, AlpacaFarm, and InjecAgent) demonstrate that DRIP outperforms state-of-the-art defenses, including StruQ, SecAlign, ISE, and PFT, improving role separation by 49%, and reducing attack success rate by 66% for adaptive attacks. Meanwhile, DRIP's utility is on par with the undefended model across AlpacaEval, IFEval, and MT-Bench. Our findings underscore the power of lightweight representation edits and role-aware supervision in securing LLMs against adaptive prompt injection.

摘要: 大型语言模型（LLM）已展示出令人印象深刻的描述跟踪能力。然而，这些功能也会使模型面临提示注入攻击，恶意制作的输入会覆盖或分散预期指令的注意力。该模型的一个核心弱点在于缺乏语义角色理解：它无法区分指令意图和描述性内容，导致它执行嵌入数据中的类似描述的短语。   我们提出了DRIP，这是一种基于语义建模角度的训练时防御，它在不牺牲实用性的情况下强制执行指令和数据语义之间的稳健分离。DRIP引入了两种轻量级但互补的机制：（1）代币式去指令转移，执行语义解纠缠，削弱数据代币中的指令语义，同时保留内容含义;（2）剩余融合路径，提供持久的语义锚点，加强真正的顶层指令在生成过程中的影响。LLaMA-8B和Mistral-7 B在三个即时注入基准（SEN、AlpacaFarm和InjecAgent）上的实验结果表明，DRIP优于最先进的防御，包括StruQ、SecAlign、ISE和PFT，将角色分离提高了49%，并将自适应攻击的攻击成功率降低了66%。与此同时，DRIP的实用性与AlpacaEval、IFEval和MT-Bench的无防御模型相当。我们的研究结果强调了轻量级表示编辑和角色感知监督在确保LLM免受自适应提示注入方面的力量。



## **33. ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training**

ToxicTextCLIP：对CLIP预训练的基于文本的中毒和后门攻击 cs.CV

Accepted by NeurIPS 2025

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00446v1) [paper-pdf](http://arxiv.org/pdf/2511.00446v1)

**Authors**: Xin Yao, Haiyang Zhao, Yimin Chen, Jiawei Guo, Kecheng Huang, Ming Zhao

**Abstract**: The Contrastive Language-Image Pretraining (CLIP) model has significantly advanced vision-language modeling by aligning image-text pairs from large-scale web data through self-supervised contrastive learning. Yet, its reliance on uncurated Internet-sourced data exposes it to data poisoning and backdoor risks. While existing studies primarily investigate image-based attacks, the text modality, which is equally central to CLIP's training, remains underexplored. In this work, we introduce ToxicTextCLIP, a framework for generating high-quality adversarial texts that target CLIP during the pre-training phase. The framework addresses two key challenges: semantic misalignment caused by background inconsistency with the target class, and the scarcity of background-consistent texts. To this end, ToxicTextCLIP iteratively applies: 1) a background-aware selector that prioritizes texts with background content aligned to the target class, and 2) a background-driven augmenter that generates semantically coherent and diverse poisoned samples. Extensive experiments on classification and retrieval tasks show that ToxicTextCLIP achieves up to 95.83% poisoning success and 98.68% backdoor Hit@1, while bypassing RoCLIP, CleanCLIP and SafeCLIP defenses. The source code can be accessed via https://github.com/xinyaocse/ToxicTextCLIP/.

摘要: 对比图像-图像预训练（CLIP）模型通过自我监督的对比学习从大规模网络数据中对齐图像-文本对，显着改进了视觉语言建模。然而，它对未经策划的互联网数据的依赖使其面临数据中毒和后门风险。虽然现有的研究主要调查基于图像的攻击，但对CLIP训练同样重要的文本模式仍然没有得到充分的研究。在这项工作中，我们引入了ToxicTextCLIP，这是一个用于在预训练阶段生成针对CLIP的高质量对抗文本的框架。该框架解决了两个关键挑战：背景与目标类不一致导致的语义失调，以及背景一致文本的稀缺。为此，ToxicTextCLIP迭代应用：1）背景感知选择器，它优先考虑背景内容与目标类对齐的文本，以及2）背景驱动增强器，它生成语义连贯且多样化的有毒样本。关于分类和检索任务的广泛实验表明，ToxicTextCLIP的中毒成功率高达95.83%，后门Hit@1达到98.68%，同时绕过RoCLIP、CleanCLIP和SafeCLIP防御。源代码可通过https://github.com/xinyaocse/ToxicTextCLIP/访问。



## **34. Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling**

通过平衡探索和利用与对象引导抽样来增强对抗性可转移性 cs.LG

accepted by iccv 2025

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00411v1) [paper-pdf](http://arxiv.org/pdf/2511.00411v1)

**Authors**: Zenghao Niu, Weicheng Xie, Siyang Song, Zitong Yu, Feng Liu, Linlin Shen

**Abstract**: Adversarial attacks present a critical challenge to deep neural networks' robustness, particularly in transfer scenarios across different model architectures. However, the transferability of adversarial attacks faces a fundamental dilemma between Exploitation (maximizing attack potency) and Exploration (enhancing cross-model generalization). Traditional momentum-based methods over-prioritize Exploitation, i.e., higher loss maxima for attack potency but weakened generalization (narrow loss surface). Conversely, recent methods with inner-iteration sampling over-prioritize Exploration, i.e., flatter loss surfaces for cross-model generalization but weakened attack potency (suboptimal local maxima). To resolve this dilemma, we propose a simple yet effective Gradient-Guided Sampling (GGS), which harmonizes both objectives through guiding sampling along the gradient ascent direction to improve both sampling efficiency and stability. Specifically, based on MI-FGSM, GGS introduces inner-iteration random sampling and guides the sampling direction using the gradient from the previous inner-iteration (the sampling's magnitude is determined by a random distribution). This mechanism encourages adversarial examples to reside in balanced regions with both flatness for cross-model generalization and higher local maxima for strong attack potency. Comprehensive experiments across multiple DNN architectures and multimodal large language models (MLLMs) demonstrate the superiority of our method over state-of-the-art transfer attacks. Code is made available at https://github.com/anuin-cat/GGS.

摘要: 对抗性攻击对深度神经网络的鲁棒性构成了严峻的挑战，特别是在跨不同模型架构的传输场景中。然而，对抗性攻击的可转移性面临着剥削（最大化攻击效力）和探索（增强跨模型概括）之间的根本困境。传统的基于动量的方法过度优先考虑剥削，即攻击效力的损失最大值更高，但概括性减弱（损失面窄）。相反，最近采用内迭代采样的方法过度优先考虑探索，即更平坦的损失表面用于跨模型概括，但攻击能力减弱（次优局部最大值）。为了解决这一困境，我们提出了一种简单而有效的样本引导采样（GGS），它通过沿着梯度上升方向引导采样来协调两个目标，以提高采样效率和稳定性。具体来说，GGS基于MI-FGSM，引入内迭代随机采样，并使用前一次内迭代的梯度引导采样方向（采样的幅度由随机分布确定）。这种机制鼓励对抗性示例驻留在平衡区域中，既具有跨模型概括的平坦性，又具有更高的局部最大值来实现强攻击效力。跨多个DNN架构和多模式大型语言模型（MLLM）的全面实验证明了我们的方法相对于最先进的传输攻击的优越性。代码可在https://github.com/anuin-cat/GGS上获取。



## **35. Stochastic Subspace Descent Accelerated via Bi-fidelity Line Search**

通过双保真线搜索加速随机子空间下降 cs.LG

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2505.00162v2) [paper-pdf](http://arxiv.org/pdf/2505.00162v2)

**Authors**: Nuojin Cheng, Alireza Doostan, Stephen Becker

**Abstract**: Efficient optimization remains a fundamental challenge across numerous scientific and engineering domains, especially when objective function and gradient evaluations are computationally expensive. While zeroth-order optimization methods offer effective approaches when gradients are inaccessible, their practical performance can be limited by the high cost associated with function queries. This work introduces the bi-fidelity stochastic subspace descent (BF-SSD) algorithm, a novel zeroth-order optimization method designed to reduce this computational burden. BF-SSD leverages a bi-fidelity framework, constructing a surrogate model from a combination of computationally inexpensive low-fidelity (LF) and accurate high-fidelity (HF) function evaluations. This surrogate model facilitates an efficient backtracking line search for step size selection, for which we provide theoretical convergence guarantees under standard assumptions. We perform a comprehensive empirical evaluation of BF-SSD across four distinct problems: a synthetic optimization benchmark, dual-form kernel ridge regression, black-box adversarial attacks on machine learning models, and transformer-based black-box language model fine-tuning. Numerical results demonstrate that BF-SSD consistently achieves superior optimization performance while requiring significantly fewer HF function evaluations compared to relevant baseline methods. This study highlights the efficacy of integrating bi-fidelity strategies within zeroth-order optimization, positioning BF-SSD as a promising and computationally efficient approach for tackling large-scale, high-dimensional problems encountered in various real-world applications.

摘要: 有效的优化仍然是众多科学和工程领域的一个根本挑战，特别是当目标函数和梯度评估计算昂贵时。虽然零阶优化方法在无法访问梯度时提供了有效的方法，但其实际性能可能会受到与函数查询相关的高成本的限制。这项工作引入了双保真随机子空间下降（BF-SSD）算法，这是一种新颖的零阶优化方法，旨在减少这种计算负担。BF-SSD利用双保真框架，从计算成本低的低保真度（LF）和准确的高保真度（HF）功能评估的组合中构建代理模型。该代理模型促进了对步骤大小选择的高效回溯线搜索，为此我们在标准假设下提供了理论收敛保证。我们针对四个不同的问题对BF-SSD进行了全面的实证评估：合成优化基准、双重形式内核岭回归、对机器学习模型的黑匣子对抗攻击以及基于转换器的黑匣子语言模型微调。数值结果表明，与相关基线方法相比，BF-SSD始终实现了卓越的优化性能，同时需要的高频功能评估显着减少。这项研究强调了在零阶优化中集成双保真策略的功效，将BF-SSD定位为一种有前途且计算效率高的方法，用于解决各种现实世界应用中遇到的大规模、多维问题。



## **36. CoP: Agentic Red-teaming for Large Language Models using Composition of Principles**

CoP：使用原则组合的大型语言模型的大型红色团队 cs.AI

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2506.00781v2) [paper-pdf](http://arxiv.org/pdf/2506.00781v2)

**Authors**: Chen Xiong, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Recent advances in Large Language Models (LLMs) have spurred transformative applications in various domains, ranging from open-source to proprietary LLMs. However, jailbreak attacks, which aim to break safety alignment and user compliance by tricking the target LLMs into answering harmful and risky responses, are becoming an urgent concern. The practice of red-teaming for LLMs is to proactively explore potential risks and error-prone instances before the release of frontier AI technology. This paper proposes an agentic workflow to automate and scale the red-teaming process of LLMs through the Composition-of-Principles (CoP) framework, where human users provide a set of red-teaming principles as instructions to an AI agent to automatically orchestrate effective red-teaming strategies and generate jailbreak prompts. Distinct from existing red-teaming methods, our CoP framework provides a unified and extensible framework to encompass and orchestrate human-provided red-teaming principles to enable the automated discovery of new red-teaming strategies. When tested against leading LLMs, CoP reveals unprecedented safety risks by finding novel jailbreak prompts and improving the best-known single-turn attack success rate by up to 19.0 times.

摘要: 大型语言模型（LLM）的最新进展激发了各个领域的变革性应用程序，从开源到专有LLM。然而，越狱攻击的目的是通过诱骗目标LLM回答有害和危险的响应来打破安全一致和用户合规性，正在成为一个紧迫的问题。LLM的红色团队实践是在前沿人工智能技术发布之前主动探索潜在风险和容易出错的实例。本文提出了一种代理工作流程，通过构成原则（CoP）框架自动化和扩展LLM的红色团队流程，其中人类用户提供一组红色团队原则作为指令，向人工智能代理自动协调有效的红色团队策略并生成越狱提示。与现有的红色团队方法不同，我们的CoP框架提供了一个统一且可扩展的框架，以涵盖和编排人类提供的红色团队原则，以实现新的红色团队策略的自动发现。当针对领先的LLM进行测试时，CoP发现了新颖的越狱提示并将最著名的单回合攻击成功率提高了19.0倍，从而揭示了前所未有的安全风险。



## **37. Exploiting Latent Space Discontinuities for Building Universal LLM Jailbreaks and Data Extraction Attacks**

利用潜在空间不连续性构建通用LLM越狱和数据提取攻击 cs.CR

10 pages, 5 figures, 4 tables, Published at the Brazilian Symposium  on Cybersecurity (SBSeg 2025)

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00346v1) [paper-pdf](http://arxiv.org/pdf/2511.00346v1)

**Authors**: Kayua Oleques Paim, Rodrigo Brandao Mansilha, Diego Kreutz, Muriel Figueredo Franco, Weverton Cordeiro

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has raised significant concerns about their security against adversarial attacks. In this work, we propose a novel approach to crafting universal jailbreaks and data extraction attacks by exploiting latent space discontinuities, an architectural vulnerability related to the sparsity of training data. Unlike previous methods, our technique generalizes across various models and interfaces, proving highly effective in seven state-of-the-art LLMs and one image generation model. Initial results indicate that when these discontinuities are exploited, they can consistently and profoundly compromise model behavior, even in the presence of layered defenses. The findings suggest that this strategy has substantial potential as a systemic attack vector.

摘要: 大型语言模型（LLM）的迅速普及引发了人们对其对抗攻击的安全性的严重担忧。在这项工作中，我们提出了一种新颖的方法，通过利用潜在空间不连续性（与训练数据稀疏性相关的架构漏洞）来制作通用越狱和数据提取攻击。与以前的方法不同，我们的技术可推广到各种模型和接口，在七种最先进的LLM和一种图像生成模型中证明非常有效。初步结果表明，当利用这些不连续性时，即使存在分层防御，它们也会持续且深刻地损害模型行为。研究结果表明，该策略具有作为系统性攻击载体的巨大潜力。



## **38. Visual Backdoor Attacks on MLLM Embodied Decision Making via Contrastive Trigger Learning**

通过对比触发学习对MLLM有序决策的视觉后门攻击 cs.AI

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.27623v1) [paper-pdf](http://arxiv.org/pdf/2510.27623v1)

**Authors**: Qiusi Zhan, Hyeonjeong Ha, Rui Yang, Sirui Xu, Hanyang Chen, Liang-Yan Gui, Yu-Xiong Wang, Huan Zhang, Heng Ji, Daniel Kang

**Abstract**: Multimodal large language models (MLLMs) have advanced embodied agents by enabling direct perception, reasoning, and planning task-oriented actions from visual inputs. However, such vision driven embodied agents open a new attack surface: visual backdoor attacks, where the agent behaves normally until a visual trigger appears in the scene, then persistently executes an attacker-specified multi-step policy. We introduce BEAT, the first framework to inject such visual backdoors into MLLM-based embodied agents using objects in the environments as triggers. Unlike textual triggers, object triggers exhibit wide variation across viewpoints and lighting, making them difficult to implant reliably. BEAT addresses this challenge by (1) constructing a training set that spans diverse scenes, tasks, and trigger placements to expose agents to trigger variability, and (2) introducing a two-stage training scheme that first applies supervised fine-tuning (SFT) and then our novel Contrastive Trigger Learning (CTL). CTL formulates trigger discrimination as preference learning between trigger-present and trigger-free inputs, explicitly sharpening the decision boundaries to ensure precise backdoor activation. Across various embodied agent benchmarks and MLLMs, BEAT achieves attack success rates up to 80%, while maintaining strong benign task performance, and generalizes reliably to out-of-distribution trigger placements. Notably, compared to naive SFT, CTL boosts backdoor activation accuracy up to 39% under limited backdoor data. These findings expose a critical yet unexplored security risk in MLLM-based embodied agents, underscoring the need for robust defenses before real-world deployment.

摘要: 多模式大型语言模型（MLLM）通过从视觉输入实现直接感知、推理和规划面向任务的动作，具有高级的具体化代理。然而，这种视觉驱动的具体代理打开了一个新的攻击面：视觉后门攻击，其中代理行为正常，直到场景中出现视觉触发，然后持续执行攻击者指定的多步策略。我们引入了BEAT，这是第一个使用环境中的对象作为触发器将此类视觉后门注入到基于MLLM的嵌入式代理中的框架。与文本触发器不同，对象触发器在视角和照明之间表现出很大的变化，因此难以可靠地植入。BEAT通过以下方式解决了这一挑战：（1）构建一个跨越不同场景、任务和触发器放置的训练集，以使代理人暴露于触发器可变性，以及（2）引入两阶段训练方案，首先应用监督式微调（SFT），然后应用我们新颖的对比触发学习（CTS）。CTS将触发区分制定为有供应商存在和无供应商输入之间的偏好学习，明确地尖锐决策边界，以确保精确的后门激活。在各种嵌入式代理基准测试和MLLM中，BEAT的攻击成功率高达80%，同时保持强大的良性任务性能，并可靠地推广到分布外触发放置。值得注意的是，与原始SFT相比，在有限的后门数据下，CTS将后门激活准确率提高至39%。这些发现暴露了基于MLLM的嵌入式代理中存在的一个严重但未探索的安全风险，强调了在现实世界部署之前需要强大的防御。



## **39. Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization**

解码LLM中的潜在攻击：通过Web摘要中的HTML提示注入 cs.CR

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2509.05831v2) [paper-pdf](http://arxiv.org/pdf/2509.05831v2)

**Authors**: Ishaan Verma

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.

摘要: 大型语言模型（LLM）越来越多地集成到基于Web的内容摘要系统中，但它们对即时注入攻击的敏感性仍然是一个紧迫的问题。在这项研究中，我们探索了如何利用非可见的HTML元素（例如<meta>、咏叹调标签和alt属性）来嵌入对抗性指令，而不改变网页的可见内容。我们引入了一个由280个静态网页组成的新颖数据集，平均分为干净和对抗注入版本，使用不同的基于HTML的策略制作。这些页面通过浏览器自动化管道进行处理，以提取原始HTML和渲染文本，密切模仿现实世界的LLM部署场景。我们评估了两个最先进的开源模型Llama 4 Scout（Meta）和Gemma 9 B IT（Google）总结此内容的能力。使用词汇（ROUGE-L）和语义（SBERT cos相似性）指标以及手动注释，我们评估这些隐蔽注入的影响。我们的研究结果显示，超过29%的注射样本导致Llama 4 Scout总结发生了显着变化，而Gemma 9 B IT的成功率较低，但并非微不足道，为15%。这些结果凸显了LLM驱动的网络管道中一个关键且在很大程度上被忽视的漏洞，其中隐藏的对抗内容可以巧妙地操纵模型输出。我们的工作为评估基于HTML的即时注入提供了一个可重复的框架和基准，并强调了涉及Web内容的LLM应用程序中对稳健的缓解策略的迫切需要。



## **40. Eliciting Secret Knowledge from Language Models**

从语言模型中提取秘密知识 cs.LG

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.01070v2) [paper-pdf](http://arxiv.org/pdf/2510.01070v2)

**Authors**: Bartosz Cywiński, Emil Ryd, Rowan Wang, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy, Samuel Marks

**Abstract**: We study secret elicitation: discovering knowledge that an AI possesses but does not explicitly verbalize. As a testbed, we train three families of large language models (LLMs) to possess specific knowledge that they apply downstream but deny knowing when asked directly. For example, in one setting, we train an LLM to generate replies that are consistent with knowing the user is female, while denying this knowledge when asked directly. We then design various black-box and white-box secret elicitation techniques and evaluate them based on whether they can help an LLM auditor successfully guess the secret knowledge. Many of our techniques improve on simple baselines. Our most effective techniques (performing best in all settings) are based on prefill attacks, a black-box technique where the LLM reveals secret knowledge when generating a completion from a predefined prefix. Our white-box techniques based on logit lens and sparse autoencoders (SAEs) also consistently increase the success rate of the LLM auditor, but are less effective. We release our models and code, establishing a public benchmark for evaluating secret elicitation methods.

摘要: 我们研究秘密启发：发现人工智能拥有但没有明确口头表达的知识。作为测试平台，我们训练三个大型语言模型（LLM）家族，使其拥有下游应用的特定知识，但在直接询问时否认知道。例如，在一种设置中，我们训练LLM生成与知道用户是女性一致的回复，而在直接询问时否认这一信息。然后，我们设计各种黑匣子和白盒秘密获取技术，并根据它们是否可以帮助LLM审核员成功猜测秘密知识来评估它们。我们的许多技术都在简单的基线上进行了改进。我们最有效的技术（在所有设置中表现最佳）基于预填充攻击，这是一种黑匣子技术，LLM在从预定义的前置生成完成时会揭示秘密知识。我们基于logit镜头和稀疏自动编码器（SAEs）的白盒技术也持续提高了LLM审核员的成功率，但效果较差。我们发布了我们的模型和代码，建立了评估秘密获取方法的公共基准。



## **41. Prevalence of Security and Privacy Risk-Inducing Usage of AI-based Conversational Agents**

安全和隐私风险的盛行--导致基于人工智能的对话代理的使用 cs.CR

10 pages, 3 figures, 5 tables, under submission

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.27275v1) [paper-pdf](http://arxiv.org/pdf/2510.27275v1)

**Authors**: Kathrin Grosse, Nico Ebert

**Abstract**: Recent improvement gains in large language models (LLMs) have lead to everyday usage of AI-based Conversational Agents (CAs). At the same time, LLMs are vulnerable to an array of threats, including jailbreaks and, for example, causing remote code execution when fed specific inputs. As a result, users may unintentionally introduce risks, for example, by uploading malicious files or disclosing sensitive information. However, the extent to which such user behaviors occur and thus potentially facilitate exploits remains largely unclear. To shed light on this issue, we surveyed a representative sample of 3,270 UK adults in 2024 using Prolific. A third of these use CA services such as ChatGPT or Gemini at least once a week. Of these ``regular users'', up to a third exhibited behaviors that may enable attacks, and a fourth have tried jailbreaking (often out of understandable reasons such as curiosity, fun or information seeking). Half state that they sanitize data and most participants report not sharing sensitive data. However, few share very sensitive data such as passwords. The majority are unaware that their data can be used to train models and that they can opt-out. Our findings suggest that current academic threat models manifest in the wild, and mitigations or guidelines for the secure usage of CAs should be developed. In areas critical to security and privacy, CAs must be equipped with effective AI guardrails to prevent, for example, revealing sensitive information to curious employees. Vendors need to increase efforts to prevent the entry of sensitive data, and to create transparency with regard to data usage policies and settings.

摘要: 大型语言模型（LLM）最近的改进导致了基于人工智能的对话代理（CA）的日常使用。与此同时，LLM容易受到一系列威胁的影响，包括越狱，例如，在输入特定输入时导致远程代码执行。因此，用户可能会无意中引入风险，例如通过上传恶意文件或披露敏感信息。然而，此类用户行为的发生程度以及可能促进漏洞利用的可能性在很大程度上仍不清楚。为了揭示这个问题，我们在2024年使用Prolific对3，270名英国成年人进行了代表性样本调查。其中三分之一的人每周至少使用一次ChatGPT或Gemini等CA服务。在这些“普通用户”中，多达三分之一的人表现出可能导致攻击的行为，四分之一的人尝试过越狱（通常是出于可以理解的原因，例如好奇心、乐趣或寻求信息）。一半的人表示他们对数据进行了清理，大多数参与者报告没有共享敏感数据。然而，很少有人共享密码等非常敏感的数据。大多数人不知道他们的数据可以用于训练模型，并且他们可以选择退出。我们的研究结果表明，当前的学术威胁模型在野外表现出来，应该制定CA安全使用的缓解措施或指南。在对安全和隐私至关重要的领域，CA必须配备有效的人工智能护栏，以防止例如向好奇的员工泄露敏感信息。供应商需要加强努力，防止敏感数据的输入，并在数据使用政策和设置方面建立透明度。



## **42. Adaptive Defense against Harmful Fine-Tuning for Large Language Models via Bayesian Data Scheduler**

通过Bayesian数据表对大型语言模型进行有害微调的自适应防御 cs.LG

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.27172v1) [paper-pdf](http://arxiv.org/pdf/2510.27172v1)

**Authors**: Zixuan Hu, Li Shen, Zhenyi Wang, Yongxian Wei, Dacheng Tao

**Abstract**: Harmful fine-tuning poses critical safety risks to fine-tuning-as-a-service for large language models. Existing defense strategies preemptively build robustness via attack simulation but suffer from fundamental limitations: (i) the infeasibility of extending attack simulations beyond bounded threat models due to the inherent difficulty of anticipating unknown attacks, and (ii) limited adaptability to varying attack settings, as simulation fails to capture their variability and complexity. To address these challenges, we propose Bayesian Data Scheduler (BDS), an adaptive tuning-stage defense strategy with no need for attack simulation. BDS formulates harmful fine-tuning defense as a Bayesian inference problem, learning the posterior distribution of each data point's safety attribute, conditioned on the fine-tuning and alignment datasets. The fine-tuning process is then constrained by weighting data with their safety attributes sampled from the posterior, thus mitigating the influence of harmful data. By leveraging the post hoc nature of Bayesian inference, the posterior is conditioned on the fine-tuning dataset, enabling BDS to tailor its defense to the specific dataset, thereby achieving adaptive defense. Furthermore, we introduce a neural scheduler based on amortized Bayesian learning, enabling efficient transfer to new data without retraining. Comprehensive results across diverse attack and defense settings demonstrate the state-of-the-art performance of our approach. Code is available at https://github.com/Egg-Hu/Bayesian-Data-Scheduler.

摘要: 有害的微调给大型语言模型的“即服务”微调带来了严重的安全风险。现有的防御策略通过攻击模拟先发制人地建立鲁棒性，但存在根本性的局限性：（i）由于预测未知攻击的固有困难，将攻击模拟扩展到有界威胁模型之外是不可行的，以及（ii）对不同攻击设置的适应性有限，因为模拟未能捕捉到它们的可变性和复杂性。为了应对这些挑战，我们提出了Bayesian Data Handler（BDS），这是一种不需要攻击模拟的自适应调整阶段防御策略。BDS将有害的微调防御制定为一个Bayesian推理问题，以微调和对齐数据集为条件，学习每个数据点安全属性的后验分布。然后，通过用从后验抽样的安全属性对数据进行加权来约束微调过程，从而减轻有害数据的影响。通过利用Bayesian推理的事后性质，后验取决于微调数据集，使BDS能够根据特定数据集定制其防御，从而实现自适应防御。此外，我们引入了基于摊销式Bayesian学习的神经调度器，无需重新训练即可高效传输到新数据。跨不同攻击和防御环境的综合结果展示了我们方法的最先进性能。代码可在https://github.com/Egg-Hu/Bayesian-Data-Scheduler上获取。



## **43. Characterizing Selective Refusal Bias in Large Language Models**

描述大型语言模型中的选择性拒绝偏差 cs.CL

21 pages, 12 figures, 14 tables

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.27087v1) [paper-pdf](http://arxiv.org/pdf/2510.27087v1)

**Authors**: Adel Khorramrouz, Sharon Levy

**Abstract**: Safety guardrails in large language models(LLMs) are developed to prevent malicious users from generating toxic content at a large scale. However, these measures can inadvertently introduce or reflect new biases, as LLMs may refuse to generate harmful content targeting some demographic groups and not others. We explore this selective refusal bias in LLM guardrails through the lens of refusal rates of targeted individual and intersectional demographic groups, types of LLM responses, and length of generated refusals. Our results show evidence of selective refusal bias across gender, sexual orientation, nationality, and religion attributes. This leads us to investigate additional safety implications via an indirect attack, where we target previously refused groups. Our findings emphasize the need for more equitable and robust performance in safety guardrails across demographic groups.

摘要: 大型语言模型（LLM）中的安全护栏旨在防止恶意用户大规模生成有毒内容。然而，这些措施可能会无意中引入或反映新的偏见，因为LLM可能会拒绝生成针对某些人口群体而不是其他群体的有害内容。我们通过目标个人和交叉人口群体的拒绝率、LLM反应类型以及产生的拒绝时间长度的视角来探讨LLM护栏中的这种选择性拒绝偏见。我们的结果显示，性别、性取向、国籍和宗教属性存在选择性拒绝偏见的证据。这导致我们通过间接攻击来调查额外的安全影响，我们的目标是之前拒绝的群体。我们的研究结果强调，各个人口群体的安全护栏需要更加公平和稳健的表现。



## **44. Adapting Large Language Models to Emerging Cybersecurity using Retrieval Augmented Generation**

使用检索增强生成使大型语言模型适应新兴网络安全 cs.CR

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.27080v1) [paper-pdf](http://arxiv.org/pdf/2510.27080v1)

**Authors**: Arnabh Borah, Md Tanvirul Alam, Nidhi Rastogi

**Abstract**: Security applications are increasingly relying on large language models (LLMs) for cyber threat detection; however, their opaque reasoning often limits trust, particularly in decisions that require domain-specific cybersecurity knowledge. Because security threats evolve rapidly, LLMs must not only recall historical incidents but also adapt to emerging vulnerabilities and attack patterns. Retrieval-Augmented Generation (RAG) has demonstrated effectiveness in general LLM applications, but its potential for cybersecurity remains underexplored. In this work, we introduce a RAG-based framework designed to contextualize cybersecurity data and enhance LLM accuracy in knowledge retention and temporal reasoning. Using external datasets and the Llama-3-8B-Instruct model, we evaluate baseline RAG, an optimized hybrid retrieval approach, and conduct a comparative analysis across multiple performance metrics. Our findings highlight the promise of hybrid retrieval in strengthening the adaptability and reliability of LLMs for cybersecurity tasks.

摘要: 安全应用程序越来越依赖大型语言模型（LLM）进行网络威胁检测;然而，它们不透明的推理通常会限制信任，特别是在需要特定领域网络安全知识的决策中。由于安全威胁迅速演变，LLM不仅必须回忆历史事件，还必须适应新出现的漏洞和攻击模式。检索增强一代（RAG）已在一般LLM应用中表现出有效性，但其网络安全潜力仍未得到充分探索。在这项工作中，我们引入了一个基于RAG的框架，旨在将网络安全数据上下文化并增强LLM知识保留和时态推理的准确性。使用外部数据集和Llama-3-8B-Instruct模型，我们评估了基线RAG，一种优化的混合检索方法，并对多个性能指标进行了比较分析。我们的研究结果强调了混合检索在加强网络安全任务LLM的适应性和可靠性方面的前景。



## **45. LLM-based Multi-class Attack Analysis and Mitigation Framework in IoT/IIoT Networks**

物联网/IIoT网络中基于LLM的多类攻击分析和缓解框架 cs.CR

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26941v1) [paper-pdf](http://arxiv.org/pdf/2510.26941v1)

**Authors**: Seif Ikbarieh, Maanak Gupta, Elmahedi Mahalal

**Abstract**: The Internet of Things has expanded rapidly, transforming communication and operations across industries but also increasing the attack surface and security breaches. Artificial Intelligence plays a key role in securing IoT, enabling attack detection, attack behavior analysis, and mitigation suggestion. Despite advancements, evaluations remain purely qualitative, and the lack of a standardized, objective benchmark for quantitatively measuring AI-based attack analysis and mitigation hinders consistent assessment of model effectiveness. In this work, we propose a hybrid framework combining Machine Learning (ML) for multi-class attack detection with Large Language Models (LLMs) for attack behavior analysis and mitigation suggestion. After benchmarking several ML and Deep Learning (DL) classifiers on the Edge-IIoTset and CICIoT2023 datasets, we applied structured role-play prompt engineering with Retrieval-Augmented Generation (RAG) to guide ChatGPT-o3 and DeepSeek-R1 in producing detailed, context-aware responses. We introduce novel evaluation metrics for quantitative assessment to guide us and an ensemble of judge LLMs, namely ChatGPT-4o, DeepSeek-V3, Mixtral 8x7B Instruct, Gemini 2.5 Flash, Meta Llama 4, TII Falcon H1 34B Instruct, xAI Grok 3, and Claude 4 Sonnet, to independently evaluate the responses. Results show that Random Forest has the best detection model, and ChatGPT-o3 outperformed DeepSeek-R1 in attack analysis and mitigation.

摘要: 物联网迅速扩张，改变了各个行业的通信和运营，但也增加了攻击面和安全漏洞。人工智能在保护物联网、实现攻击检测、攻击行为分析和缓解建议方面发挥着关键作用。尽管取得了进步，但评估仍然纯粹是定性的，缺乏用于定量测量基于人工智能的攻击分析和缓解的标准化、客观的基准阻碍了对模型有效性的一致评估。在这项工作中，我们提出了一个混合框架，将用于多类攻击检测的机器学习（ML）与用于攻击行为分析和缓解建议的大型语言模型（LLM）相结合。在Edge-IIoTset和CICIoT 2023数据集上对几个ML和深度学习（DL）分类器进行基准测试后，我们应用了具有检索增强生成（RAG）的结构化角色扮演提示工程来指导ChatGPT-o3和DeepSeek-R1生成详细的上下文感知响应。我们引入了新颖的定量评估指标来指导我们和一系列法官LLM，即ChatGPT-4 o、DeepSeek-V3、Mixtral 8x 7 B Direct、Gemini 2.5 Flash、Meta Llama 4、TII Falcon H1 34 B Direct、xAI Grok 3和Claude 4十四行诗，以独立评估响应。结果表明，Random Forest具有最好的检测模型，ChatGPT-o3在攻击分析和缓解方面优于DeepSeek-R1。



## **46. LatentBreak: Jailbreaking Large Language Models through Latent Space Feedback**

LatentBreak：通过潜在空间反馈越狱大型语言模型 cs.CL

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.08604v2) [paper-pdf](http://arxiv.org/pdf/2510.08604v2)

**Authors**: Raffaele Mura, Giorgio Piras, Kamilė Lukošiūtė, Maura Pintor, Amin Karbasi, Battista Biggio

**Abstract**: Jailbreaks are adversarial attacks designed to bypass the built-in safety mechanisms of large language models. Automated jailbreaks typically optimize an adversarial suffix or adapt long prompt templates by forcing the model to generate the initial part of a restricted or harmful response. In this work, we show that existing jailbreak attacks that leverage such mechanisms to unlock the model response can be detected by a straightforward perplexity-based filtering on the input prompt. To overcome this issue, we propose LatentBreak, a white-box jailbreak attack that generates natural adversarial prompts with low perplexity capable of evading such defenses. LatentBreak substitutes words in the input prompt with semantically-equivalent ones, preserving the initial intent of the prompt, instead of adding high-perplexity adversarial suffixes or long templates. These words are chosen by minimizing the distance in the latent space between the representation of the adversarial prompt and that of harmless requests. Our extensive evaluation shows that LatentBreak leads to shorter and low-perplexity prompts, thus outperforming competing jailbreak algorithms against perplexity-based filters on multiple safety-aligned models.

摘要: 越狱是旨在绕过大型语言模型内置安全机制的对抗性攻击。自动越狱通常会通过强制模型生成受限或有害响应的初始部分来优化对抗性后缀或调整长提示模板。在这项工作中，我们表明，利用此类机制解锁模型响应的现有越狱攻击可以通过对输入提示进行简单的基于困惑的过滤来检测。为了克服这个问题，我们提出了LatentBreak，这是一种白盒越狱攻击，可以生成具有低困惑度的自然对抗提示，能够逃避此类防御。LatentBreak将输入提示中的单词替换为语义等效的单词，保留提示的初始意图，而不是添加高困惑度的对抗性后缀或长模板。这些词是通过最小化潜在空间中对抗性提示的表示与无害请求的表示之间的距离来选择的。我们广泛的评估表明，LatentBreak导致更短和低困惑的提示，从而优于竞争的越狱算法对基于困惑的过滤器在多个安全对齐的模型。



## **47. Broken-Token: Filtering Obfuscated Prompts by Counting Characters-Per-Token**

破碎的令牌：通过计算每个令牌的数量来过滤混淆的令牌 cs.CR

16 pages, 9 figures

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26847v1) [paper-pdf](http://arxiv.org/pdf/2510.26847v1)

**Authors**: Shaked Zychlinski, Yuval Kainan

**Abstract**: Large Language Models (LLMs) are susceptible to jailbreak attacks where malicious prompts are disguised using ciphers and character-level encodings to bypass safety guardrails. While these guardrails often fail to interpret the encoded content, the underlying models can still process the harmful instructions. We introduce CPT-Filtering, a novel, model-agnostic with negligible-costs and near-perfect accuracy guardrail technique that aims to mitigate these attacks by leveraging the intrinsic behavior of Byte-Pair Encoding (BPE) tokenizers. Our method is based on the principle that tokenizers, trained on natural language, represent out-of-distribution text, such as ciphers, using a significantly higher number of shorter tokens. Our technique uses a simple yet powerful artifact of using language models: the average number of Characters Per Token (CPT) in the text. This approach is motivated by the high compute cost of modern methods - relying on added modules such as dedicated LLMs or perplexity models. We validate our approach across a large dataset of over 100,000 prompts, testing numerous encoding schemes with several popular tokenizers. Our experiments demonstrate that a simple CPT threshold robustly identifies encoded text with high accuracy, even for very short inputs. CPT-Filtering provides a practical defense layer that can be immediately deployed for real-time text filtering and offline data curation.

摘要: 大型语言模型（LLM）很容易受到越狱攻击，其中恶意提示使用密码和字符级编码来伪装以绕过安全护栏。虽然这些护栏通常无法解释编码内容，但底层模型仍然可以处理有害指令。我们引入了CPT-Filting，这是一种新颖的、模型不可知的护栏技术，成本可忽略不计，准确度近乎完美，旨在通过利用字节对编码（BPE）标记器的内在行为来减轻这些攻击。我们的方法基于这样的原则：在自然语言上训练的符号化器，使用数量明显较多的较短符号来表示分发外文本，例如密码。我们的技术使用了一个简单但强大的语言模型人工制品：文本中每个令牌的平均字符数（CPD）。这种方法的动机是现代方法的高计算成本--依赖于添加的模块，例如专用LLM或困惑模型。我们在包含超过100，000个提示的大型数据集中验证了我们的方法，并使用几种流行的标记器测试了众多编码方案。我们的实验表明，简单的CPD阈值能够以高准确性稳健地识别编码文本，即使对于非常短的输入也是如此。CPD过滤提供了一个实用的防御层，可以立即部署用于实时文本过滤和离线数据策展。



## **48. PVMark: Enabling Public Verifiability for LLM Watermarking Schemes**

PVMark：实现LLM水印计划的公共可验证性 cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26274v1) [paper-pdf](http://arxiv.org/pdf/2510.26274v1)

**Authors**: Haohua Duan, Liyao Xiang, Xin Zhang

**Abstract**: Watermarking schemes for large language models (LLMs) have been proposed to identify the source of the generated text, mitigating the potential threats emerged from model theft. However, current watermarking solutions hardly resolve the trust issue: the non-public watermark detection cannot prove itself faithfully conducting the detection. We observe that it is attributed to the secret key mostly used in the watermark detection -- it cannot be public, or the adversary may launch removal attacks provided the key; nor can it be private, or the watermarking detection is opaque to the public. To resolve the dilemma, we propose PVMark, a plugin based on zero-knowledge proof (ZKP), enabling the watermark detection process to be publicly verifiable by third parties without disclosing any secret key. PVMark hinges upon the proof of `correct execution' of watermark detection on which a set of ZKP constraints are built, including mapping, random number generation, comparison, and summation. We implement multiple variants of PVMark in Python, Rust and Circom, covering combinations of three watermarking schemes, three hash functions, and four ZKP protocols, to show our approach effectively works under a variety of circumstances. By experimental results, PVMark efficiently enables public verifiability on the state-of-the-art LLM watermarking schemes yet without compromising the watermarking performance, promising to be deployed in practice.

摘要: 人们提出了大型语言模型（LLM）的水印方案来识别生成文本的来源，从而减轻模型盗窃带来的潜在威胁。然而，当前的水印解决方案很难解决信任问题：非公开水印检测无法证明自己忠实地进行检测。我们观察到，它归因于水印检测中大多使用的密钥--它不能是公开的，否则对手可能会在提供密钥的情况下发起删除攻击;它也不能是私人的，或者水印检测对公众不透明。为了解决这个困境，我们提出了PVMark，这是一个基于零知识证明（ZKP）的插件，使水印检测过程能够由第三方公开验证，而无需披露任何密钥。PVMark取决于水印检测“正确执行”的证明，在此基础上构建了一组ZKP约束，包括映射、随机数生成、比较和总和。我们在Python、Rust和Circom中实现了PVMark的多个变体，涵盖了三种水印方案、三种哈希函数和四种ZKP协议的组合，以表明我们的方法在各种情况下有效工作。根据实验结果，PVMark有效地实现了最先进的LLM水印方案的公开验证性，同时又不损害水印性能，有望在实践中部署。



## **49. IRCopilot: Automated Incident Response with Large Language Models**

IRCopilot：使用大型语言模型的自动事件响应 cs.CR

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2505.20945v3) [paper-pdf](http://arxiv.org/pdf/2505.20945v3)

**Authors**: Xihuan Lin, Jie Zhang, Gelei Deng, Tianzhe Liu, Tianwei Zhang, Qing Guo, Riqing Chen

**Abstract**: Incident response plays a pivotal role in mitigating the impact of cyber attacks. In recent years, the intensity and complexity of global cyber threats have grown significantly, making it increasingly challenging for traditional threat detection and incident response methods to operate effectively in complex network environments. While Large Language Models (LLMs) have shown great potential in early threat detection, their capabilities remain limited when it comes to automated incident response after an intrusion. To address this gap, we construct an incremental benchmark based on real-world incident response tasks to thoroughly evaluate the performance of LLMs in this domain. Our analysis reveals several key challenges that hinder the practical application of contemporary LLMs, including context loss, hallucinations, privacy protection concerns, and their limited ability to provide accurate, context-specific recommendations. In response to these challenges, we propose IRCopilot, a novel framework for automated incident response powered by LLMs. IRCopilot mimics the three dynamic phases of a real-world incident response team using four collaborative LLM-based session components. These components are designed with clear divisions of responsibility, reducing issues such as hallucinations and context loss. Our method leverages diverse prompt designs and strategic responsibility segmentation, significantly improving the system's practicality and efficiency. Experimental results demonstrate that IRCopilot outperforms baseline LLMs across key benchmarks, achieving sub-task completion rates of 150%, 138%, 136%, 119%, and 114% for various response tasks. Moreover, IRCopilot exhibits robust performance on public incident response platforms and in real-world attack scenarios, showcasing its strong applicability.

摘要: 事件响应在减轻网络攻击的影响方面发挥着关键作用。近年来，全球网络威胁的强度和复杂性显着增长，使得传统的威胁检测和事件响应方法在复杂网络环境中有效运作面临越来越大的挑战。虽然大型语言模型（LLM）在早期威胁检测方面表现出了巨大的潜力，但在入侵后自动化事件响应方面，它们的能力仍然有限。为了解决这一差距，我们基于现实世界的事件响应任务构建了一个增量基准，以彻底评估LLM在该领域的性能。我们的分析揭示了阻碍当代LLM实际应用的几个关键挑战，包括上下文丢失、幻觉、隐私保护问题，以及它们提供准确的、针对特定上下文的建议的能力有限。为了应对这些挑战，我们提出了IRCopilot，这是一个由LLM支持的自动化事件响应的新型框架。IRCopilot使用四个基于LLM的协作会话组件模拟现实世界事件响应团队的三个动态阶段。这些组件的设计有明确的责任分工，减少了幻觉和上下文丢失等问题。我们的方法利用多样化的提示设计和战略责任细分，显着提高了系统的实用性和效率。实验结果表明，IRCopilot在关键基准上的表现优于基线LLM，各种响应任务的子任务完成率分别为150%、138%、136%、119%和114%。此外，IRCopilot在公共事件响应平台和现实世界的攻击场景中表现出稳健的性能，展示了其强大的适用性。



## **50. ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models**

ALMGGuard：安全捷径以及在哪里找到它们作为音频语言模型的护栏 cs.SD

Accepted to NeurIPS 2025

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26096v1) [paper-pdf](http://arxiv.org/pdf/2510.26096v1)

**Authors**: Weifei Jin, Yuxin Cao, Junjie Su, Minhui Xue, Jie Hao, Ke Xu, Jin Song Dong, Derui Wang

**Abstract**: Recent advances in Audio-Language Models (ALMs) have significantly improved multimodal understanding capabilities. However, the introduction of the audio modality also brings new and unique vulnerability vectors. Previous studies have proposed jailbreak attacks that specifically target ALMs, revealing that defenses directly transferred from traditional audio adversarial attacks or text-based Large Language Model (LLM) jailbreaks are largely ineffective against these ALM-specific threats. To address this issue, we propose ALMGuard, the first defense framework tailored to ALMs. Based on the assumption that safety-aligned shortcuts naturally exist in ALMs, we design a method to identify universal Shortcut Activation Perturbations (SAPs) that serve as triggers that activate the safety shortcuts to safeguard ALMs at inference time. To better sift out effective triggers while preserving the model's utility on benign tasks, we further propose Mel-Gradient Sparse Mask (M-GSM), which restricts perturbations to Mel-frequency bins that are sensitive to jailbreaks but insensitive to speech understanding. Both theoretical analyses and empirical results demonstrate the robustness of our method against both seen and unseen attacks. Overall, \MethodName reduces the average success rate of advanced ALM-specific jailbreak attacks to 4.6% across four models, while maintaining comparable utility on benign benchmarks, establishing it as the new state of the art. Our code and data are available at https://github.com/WeifeiJin/ALMGuard.

摘要: 音频语言模型（ILM）的最新进展显着提高了多模式理解能力。然而，音频模式的引入也带来了新的独特漏洞载体。之前的研究提出了专门针对ILM的越狱攻击，揭示了直接从传统音频对抗攻击或基于文本的大型语言模型（LLM）越狱转移的防御对于这些ILM特定的威胁基本上无效。为了解决这个问题，我们提出了ALMGGuard，这是第一个针对ILM量身定制的防御框架。基于APM中自然存在安全对齐快捷方式这一假设，我们设计了一种方法来识别通用收件箱激活扰动（SAP），该扰动充当触发器，在推理时激活安全快捷方式以保护APM。为了更好地筛选出有效的触发器，同时保留模型对良性任务的实用性，我们进一步提出了Mel梯度稀疏屏蔽（M-GSM），它将扰动限制在对越狱敏感但对语音理解不敏感的Mel频率箱。理论分析和经验结果都证明了我们的方法对可见和不可见的攻击的鲁棒性。总体而言，\MethodName将四种模型中高级ILM特定越狱攻击的平均成功率降低至4.6%，同时在良性基准上保持了相当的实用性，使其成为最新的最新技术水平。我们的代码和数据可在https://github.com/WeifeiJin/ALMGuard上获取。



