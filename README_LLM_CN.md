# Latest Large Language Model Attack Papers
**update at 2025-10-30 15:01:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Securing AI Agent Execution**

确保AI代理执行 cs.CR

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.21236v2) [paper-pdf](http://arxiv.org/pdf/2510.21236v2)

**Authors**: Christoph Bühler, Matteo Biagiola, Luca Di Grazia, Guido Salvaneschi

**Abstract**: Large Language Models (LLMs) have evolved into AI agents that interact with external tools and environments to perform complex tasks. The Model Context Protocol (MCP) has become the de facto standard for connecting agents with such resources, but security has lagged behind: thousands of MCP servers execute with unrestricted access to host systems, creating a broad attack surface. In this paper, we introduce AgentBound, the first access control framework for MCP servers. AgentBound combines a declarative policy mechanism, inspired by the Android permission model, with a policy enforcement engine that contains malicious behavior without requiring MCP server modifications. We build a dataset containing the 296 most popular MCP servers, and show that access control policies can be generated automatically from source code with 80.9% accuracy. We also show that AgentBound blocks the majority of security threats in several malicious MCP servers, and that policy enforcement engine introduces negligible overhead. Our contributions provide developers and project managers with a practical foundation for securing MCP servers while maintaining productivity, enabling researchers and tool builders to explore new directions for declarative access control and MCP security.

摘要: 大型语言模型（LLM）已发展成为与外部工具和环境交互以执行复杂任务的人工智能代理。模型上下文协议（HCP）已成为连接代理与此类资源的事实上的标准，但安全性却落后：数千个LCP服务器在不受限制地访问主机系统的情况下执行，从而造成了广泛的攻击面。本文中，我们介绍了AgentBound，这是第一个针对LCP服务器的访问控制框架。AgentBound将受Android权限模型启发的声明性策略机制与包含恶意行为而无需修改LCP服务器的策略执行引擎相结合。我们构建了一个包含296个最流行的LCP服务器的数据集，并表明访问控制策略可以从源代码自动生成，准确率为80.9%。我们还表明，AgentBound可以阻止多个恶意LCP服务器中的大部分安全威胁，并且策略执行引擎引入的额外费用可以忽略不计。我们的贡献为开发人员和项目经理提供了在保持生产力的同时保护LCP服务器的实用基础，使研究人员和工具构建者能够探索声明性访问控制和LCP安全的新方向。



## **2. NetEcho: From Real-World Streaming Side-Channels to Full LLM Conversation Recovery**

NetEcho：从现实世界的流媒体副频道到完整的LLM对话恢复 cs.CR

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.25472v1) [paper-pdf](http://arxiv.org/pdf/2510.25472v1)

**Authors**: Zheng Zhang, Guanlong Wu, Sen Deng, Shuai Wang, Yinqian Zhang

**Abstract**: In the rapidly expanding landscape of Large Language Model (LLM) applications, real-time output streaming has become the dominant interaction paradigm. While this enhances user experience, recent research reveals that it exposes a non-trivial attack surface through network side-channels. Adversaries can exploit patterns in encrypted traffic to infer sensitive information and reconstruct private conversations. In response, LLM providers and third-party services are deploying defenses such as traffic padding and obfuscation to mitigate these vulnerabilities.   This paper starts by presenting a systematic analysis of contemporary side-channel defenses in mainstream LLM applications, with a focus on services from vendors like OpenAI and DeepSeek. We identify and examine seven representative deployment scenarios, each incorporating active/passive mitigation techniques. Despite these enhanced security measures, our investigation uncovers significant residual information that remains vulnerable to leakage within the network traffic.   Building on this discovery, we introduce NetEcho, a novel, LLM-based framework that comprehensively unleashes the network side-channel risks of today's LLM applications. NetEcho is designed to recover entire conversations -- including both user prompts and LLM responses -- directly from encrypted network traffic. It features a deliberate design that ensures high-fidelity text recovery, transferability across different deployment scenarios, and moderate operational cost. In our evaluations on medical and legal applications built upon leading models like DeepSeek-v3 and GPT-4o, NetEcho can recover avg $\sim$70\% information of each conversation, demonstrating a critical limitation in current defense mechanisms. We conclude by discussing the implications of our findings and proposing future directions for augmenting network traffic security.

摘要: 在大型语言模型（LLM）应用程序的迅速扩大中，实时输出流已成为主导的交互范式。虽然这增强了用户体验，但最近的研究表明，它通过网络侧渠道暴露了一个不平凡的攻击表面。对手可以利用加密流量中的模式来推断敏感信息并重建私人对话。作为回应，LLM提供商和第三方服务正在部署流量填充和混淆等防御措施来缓解这些漏洞。   本文首先对主流LLM应用程序中的当代侧通道防御进行了系统分析，重点关注OpenAI和DeepSeek等供应商的服务。我们确定并检查了七种代表性的部署场景，每种场景都结合了主动/被动缓解技术。尽管采取了这些增强的安全措施，我们的调查仍发现了大量残留信息，这些信息仍然容易在网络流量中泄露。   在这一发现的基础上，我们引入了NetEcho，这是一个基于LLM的新颖框架，可以全面释放当今LLM应用程序的网络侧通道风险。NetEcho旨在直接从加密的网络流量中恢复整个对话（包括用户提示和LLM响应）。它采用精心设计的设计，确保高保真文本恢复、不同部署场景之间的可移植性以及适度的运营成本。在我们对基于DeepSeek-v3和GPT-4 o等领先模型的医疗和法律应用程序的评估中，NetEcho可以恢复每次对话的平均$70%信息，这表明了当前防御机制的严重局限性。最后，我们讨论了我们研究结果的影响，并提出了增强网络流量安全的未来方向。



## **3. Pentest-R1: Towards Autonomous Penetration Testing Reasoning Optimized via Two-Stage Reinforcement Learning**

Pentest-R1：通过两阶段强化学习优化自主渗透测试推理 cs.AI

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2508.07382v2) [paper-pdf](http://arxiv.org/pdf/2508.07382v2)

**Authors**: He Kong, Die Hu, Jingguo Ge, Liangxiong Li, Hui Li, Tong Li

**Abstract**: Automating penetration testing is crucial for enhancing cybersecurity, yet current Large Language Models (LLMs) face significant limitations in this domain, including poor error handling, inefficient reasoning, and an inability to perform complex end-to-end tasks autonomously. To address these challenges, we introduce Pentest-R1, a novel framework designed to optimize LLM reasoning capabilities for this task through a two-stage reinforcement learning pipeline. We first construct a dataset of over 500 real-world, multi-step walkthroughs, which Pentest-R1 leverages for offline reinforcement learning (RL) to instill foundational attack logic. Subsequently, the LLM is fine-tuned via online RL in an interactive Capture The Flag (CTF) environment, where it learns directly from environmental feedback to develop robust error self-correction and adaptive strategies. Our extensive experiments on the Cybench and AutoPenBench benchmarks demonstrate the framework's effectiveness. On AutoPenBench, Pentest-R1 achieves a 24.2\% success rate, surpassing most state-of-the-art models and ranking second only to Gemini 2.5 Flash. On Cybench, it attains a 15.0\% success rate in unguided tasks, establishing a new state-of-the-art for open-source LLMs and matching the performance of top proprietary models. Ablation studies confirm that the synergy of both training stages is critical to its success.

摘要: 自动化渗透测试对于增强网络安全至关重要，但当前的大型语言模型（LLM）在该领域面临着显着的局限性，包括错误处理不良、推理效率低下以及无法自主执行复杂的端到端任务。为了应对这些挑战，我们引入了Pentest-R1，这是一个新颖的框架，旨在通过两阶段强化学习管道优化该任务的LLM推理能力。我们首先构建了一个包含500多个现实世界的多步骤步行表的数据集，Pentest-R1利用该数据集进行离线强化学习（RL）来灌输基础攻击逻辑。随后，LLM在交互式捕获旗帜（CTF）环境中通过在线RL进行微调，直接从环境反馈中学习，以制定稳健的错误自我纠正和自适应策略。我们对Cybank和AutoPenBench基准的广泛实验证明了该框架的有效性。在AutoPenBench上，Pentest-R1的成功率达到了24.2%，超越了大多数最先进的型号，仅次于Gemini 2.5 Flash。在Cybank上，它在无引导任务中的成功率达到了15.0%，为开源LLM建立了新的最先进技术，并与顶级专有模型的性能相匹配。消融研究证实，两个训练阶段的协同作用对其成功至关重要。



## **4. Agentic Moderation: Multi-Agent Design for Safer Vision-Language Models**

智能调节：更安全视觉语言模型的多智能体设计 cs.AI

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.25179v1) [paper-pdf](http://arxiv.org/pdf/2510.25179v1)

**Authors**: Juan Ren, Mark Dras, Usman Naseem

**Abstract**: Agentic methods have emerged as a powerful and autonomous paradigm that enhances reasoning, collaboration, and adaptive control, enabling systems to coordinate and independently solve complex tasks. We extend this paradigm to safety alignment by introducing Agentic Moderation, a model-agnostic framework that leverages specialised agents to defend multimodal systems against jailbreak attacks. Unlike prior approaches that apply as a static layer over inputs or outputs and provide only binary classifications (safe or unsafe), our method integrates dynamic, cooperative agents, including Shield, Responder, Evaluator, and Reflector, to achieve context-aware and interpretable moderation. Extensive experiments across five datasets and four representative Large Vision-Language Models (LVLMs) demonstrate that our approach reduces the Attack Success Rate (ASR) by 7-19%, maintains a stable Non-Following Rate (NF), and improves the Refusal Rate (RR) by 4-20%, achieving robust, interpretable, and well-balanced safety performance. By harnessing the flexibility and reasoning capacity of agentic architectures, Agentic Moderation provides modular, scalable, and fine-grained safety enforcement, highlighting the broader potential of agentic systems as a foundation for automated safety governance.

摘要: 抽象方法已成为一种强大且自主的范式，可以增强推理、协作和自适应控制，使系统能够协调和独立解决复杂任务。我们通过引入显式调节来将这个范式扩展到安全一致，显式调节是一个模型不可知的框架，利用专门的代理来保护多模式系统免受越狱攻击。与以前的方法不同，这些方法应用于输入或输出上的静态层，并且仅提供二进制分类（安全或不安全），我们的方法集成了动态、合作的代理，包括Shield、Responder、Everator和Reflector，以实现上下文感知和可解释的审核。跨五个数据集和四个代表性的大型视觉语言模型（LVLM）的广泛实验表明，我们的方法将攻击成功率（ASB）降低7- 19%，保持稳定的非跟随率（NF），并将拒绝率（RR）提高4- 20%，实现稳健、可解释且平衡良好的安全性能。通过利用代理体系结构的灵活性和推理能力，扩展调节提供模块化、可扩展和细粒度的安全执行，凸显了代理系统作为自动化安全治理基础的更广泛潜力。



## **5. OpenGuardrails: A Configurable, Unified, and Scalable Guardrails Platform for Large Language Models**

OpenGuardrails：用于大型语言模型的可配置、统一和可扩展的Guardrails平台 cs.CR

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.19169v2) [paper-pdf](http://arxiv.org/pdf/2510.19169v2)

**Authors**: Thomas Wang, Haowen Li

**Abstract**: As large language models (LLMs) are increasingly integrated into real-world applications, ensuring their safety, robustness, and privacy compliance has become critical. We present OpenGuardrails, the first fully open-source platform that unifies large-model-based safety detection, manipulation defense, and deployable guardrail infrastructure. OpenGuardrails protects against three major classes of risks: (1) content-safety violations such as harmful or explicit text generation, (2) model-manipulation attacks including prompt injection, jailbreaks, and code-interpreter abuse, and (3) data leakage involving sensitive or private information. Unlike prior modular or rule-based frameworks, OpenGuardrails introduces three core innovations: (1) a Configurable Policy Adaptation mechanism that allows per-request customization of unsafe categories and sensitivity thresholds; (2) a Unified LLM-based Guard Architecture that performs both content-safety and manipulation detection within a single model; and (3) a Quantized, Scalable Model Design that compresses a 14B dense base model to 3.3B via GPTQ while preserving over 98 of benchmark accuracy. The system supports 119 languages, achieves state-of-the-art performance across multilingual safety benchmarks, and can be deployed as a secure gateway or API-based service for enterprise use. All models, datasets, and deployment scripts are released under the Apache 2.0 license.

摘要: 随着大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，确保其安全性、稳健性和隐私合规性变得至关重要。我们展示了OpenGuardrails，这是第一个完全开源的平台，它统一了基于大型模型的安全检测、操纵防御和可部署的护栏基础设施。OpenGuardrails可防范三类主要风险：（1）内容安全违规，例如有害或显式文本生成，（2）模型操纵攻击，包括提示注入、越狱和代码解释器滥用，以及（3）涉及敏感或私人信息的数据泄露。与之前的模块化或基于规则的框架不同，OpenGuardrails引入了三个核心创新：（1）可配置策略适应机制，允许按请求自定义不安全类别和敏感度阈值;（2）基于统一LLM的Guard架构，在单个模型内执行内容安全和操纵检测;和（3）量化、可扩展的模型设计，通过GPTQ将14 B密集基础模型压缩到3.3B，同时保持超过98的基准准确性。该系统支持119种语言，在多语言安全基准中实现了最先进的性能，并且可以部署为安全网关或基于API的服务供企业使用。所有模型、数据集和部署脚本均在Apache 2.0许可下发布。



## **6. Secure Retrieval-Augmented Generation against Poisoning Attacks**

针对中毒攻击的安全检索增强生成 cs.CR

To appear in IEEE BigData 2025

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.25025v1) [paper-pdf](http://arxiv.org/pdf/2510.25025v1)

**Authors**: Zirui Cheng, Jikai Sun, Anjun Gao, Yueyang Quan, Zhuqing Liu, Xiaohua Hu, Minghong Fang

**Abstract**: Large language models (LLMs) have transformed natural language processing (NLP), enabling applications from content generation to decision support. Retrieval-Augmented Generation (RAG) improves LLMs by incorporating external knowledge but also introduces security risks, particularly from data poisoning, where the attacker injects poisoned texts into the knowledge database to manipulate system outputs. While various defenses have been proposed, they often struggle against advanced attacks. To address this, we introduce RAGuard, a detection framework designed to identify poisoned texts. RAGuard first expands the retrieval scope to increase the proportion of clean texts, reducing the likelihood of retrieving poisoned content. It then applies chunk-wise perplexity filtering to detect abnormal variations and text similarity filtering to flag highly similar texts. This non-parametric approach enhances RAG security, and experiments on large-scale datasets demonstrate its effectiveness in detecting and mitigating poisoning attacks, including strong adaptive attacks.

摘要: 大型语言模型（LLM）改变了自然语言处理（NLP），使从内容生成到决策支持的应用程序成为可能。检索增强生成（RAG）通过整合外部知识来改进LLM，但也会引入安全风险，特别是来自数据中毒的风险，即攻击者将有毒文本注入知识数据库以操纵系统输出。虽然已经提出了各种防御措施，但它们常常难以抵御高级攻击。为了解决这个问题，我们引入了RAGuard，这是一个旨在识别有毒文本的检测框架。RAGuard首先扩大检索范围，增加干净文本的比例，降低检索有毒内容的可能性。然后，它应用块式困惑过滤来检测异常变化，并应用文本相似性过滤来标记高度相似的文本。这种非参数方法增强了RAG安全性，大规模数据集上的实验证明了其在检测和减轻中毒攻击（包括强适应性攻击）方面的有效性。



## **7. S3C2 Summit 2025-03: Industry Secure Supply Chain Summit**

S3 C2峰会2025-03：行业安全供应链峰会 cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24920v1) [paper-pdf](http://arxiv.org/pdf/2510.24920v1)

**Authors**: Elizabeth Lin, Jonah Ghebremichael, William Enck, Yasemin Acar, Michel Cukier, Alexandros Kapravelos, Christian Kastner, Laurie Williams

**Abstract**: Software supply chains, while providing immense economic and software development value, are only as strong as their weakest link. Over the past several years, there has been an exponential increase in cyberattacks specifically targeting vulnerable links in critical software supply chains. These attacks disrupt the day-to-day functioning and threaten the security of nearly everyone on the internet, from billion-dollar companies and government agencies to hobbyist open-source developers. The ever-evolving threat of software supply chain attacks has garnered interest from both the software industry and US government in improving software supply chain security. On Thursday, March 6th, 2025, four researchers from the NSF-backed Secure Software Supply Chain Center (S3C2) conducted a Secure Software Supply Chain Summit with a diverse set of 18 practitioners from 17 organizations. The goals of the Summit were: (1) to enable sharing between participants from different industries regarding practical experiences and challenges with software supply chain security; (2) to help form new collaborations; and (3) to learn about the challenges facing participants to inform our future research directions. The summit consisted of discussions of six topics relevant to the government agencies represented, including software bill of materials (SBOMs); compliance; malicious commits; build infrastructure; culture; and large language models (LLMs) and security. For each topic of discussion, we presented a list of questions to participants to spark conversation. In this report, we provide a summary of the summit. The open questions and challenges that remained after each topic are listed at the end of each topic's section, and the initial discussion questions for each topic are provided in the appendix.

摘要: 软件供应链虽然提供了巨大的经济和软件开发价值，但其最薄弱的环节的强大程度取决于其最薄弱的环节。在过去的几年里，专门针对关键软件供应链中脆弱环节的网络攻击呈指数级增加。这些攻击扰乱了日常运作，并威胁到互联网上几乎所有人的安全，从价值数十亿美元的公司和政府机构到爱好者的开源开发人员。软件供应链攻击的不断变化的威胁引起了软件行业和美国政府对提高软件供应链安全性的兴趣。2025年3月6日，星期四，来自NSF支持的安全软件供应链中心（S3 C2）的四名研究人员与来自17个组织的18名从业者进行了安全软件供应链峰会。峰会的目的是：（1）让来自不同行业的参与者分享软件供应链安全的实践经验和挑战;（2）帮助形成新的合作;以及（3）了解参与者面临的挑战，以告知我们未来的研究方向。峰会讨论了与政府机构相关的六个主题，包括软件物料清单（SBOM）;合规性;恶意提交;构建基础设施;文化;大型语言模型（LLM）和安全性。对于每个讨论主题，我们向参与者提出了一系列问题，以激发对话。在本报告中，我们提供了峰会摘要。每个主题之后剩余的开放问题和挑战列在每个主题部分的结尾，每个主题的初步讨论问题列在附录中。



## **8. Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs**

Video-SafetyBench：视频LVLM安全评估的基准 cs.CV

Accepted by NeurIPS 2025 Dataset and Benchmark Track, Project page:  https://liuxuannan.github.io/Video-SafetyBench.github.io/

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2505.11842v3) [paper-pdf](http://arxiv.org/pdf/2505.11842v3)

**Authors**: Xuannan Liu, Zekun Li, Zheqi He, Peipei Li, Shuhan Xia, Xing Cui, Huaibo Huang, Xi Yang, Ran He

**Abstract**: The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies.

摘要: 大型视觉语言模型（LVLM）的越来越多的部署引发了潜在恶意输入下的安全问题。然而，现有的多模式安全评估主要关注静态图像输入暴露的模型漏洞，而忽略了可能引发明显安全风险的视频的时间动态。为了弥合这一差距，我们引入了Video-SafetyBench，这是第一个旨在评估LVLM在视频文本攻击下的安全性的综合基准。它由2，264个视频-文本对组成，涵盖48个细粒度的不安全类别，每个将合成视频与包含明显恶意的有害查询或良性查询配对，良性查询看似无害，但在与视频一起解释时会触发有害行为。为了生成语义准确的视频以进行安全评估，我们设计了一个可控的管道，将视频语义分解为主题图像（显示的内容）和运动文本（它如何移动），共同指导查询相关视频的合成。为了有效地评估不确定或边缘有害输出，我们提出了RJScore，这是一种新型的基于LLM的指标，它结合了判断模型的置信度和人类一致的决策阈值校准。大量实验表明，良性查询视频合成的平均攻击成功率为67.2%，揭示了视频诱导攻击的一致漏洞。我们相信Video-SafetyBench将促进未来对基于视频的安全评估和防御策略的研究。



## **9. MixAT: Combining Continuous and Discrete Adversarial Training for LLMs**

MixAT：结合LLM的连续和离散对抗训练 cs.LG

Published at 39th Conference on Neural Information Processing Systems  (NeurIPS 2025)

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2505.16947v2) [paper-pdf](http://arxiv.org/pdf/2505.16947v2)

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: Despite recent efforts in Large Language Model (LLM) safety and alignment, current adversarial attacks on frontier LLMs can still consistently force harmful generations. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. At the same time, despite their effectiveness and generalization capabilities, training with continuous perturbations does not always capture the full spectrum of vulnerabilities exploited by discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.

摘要: 尽管最近在大型语言模型（LLM）的安全性和对齐方面做出了努力，但目前对前沿LLM的对抗性攻击仍然可以持续地迫使有害的世代。尽管对抗性训练已经被广泛研究，并被证明可以显着提高传统机器学习模型的鲁棒性，但它在LLM背景下的优势和劣势却鲜为人知。具体来说，虽然现有的离散对抗性攻击在产生有害内容方面是有效的，但用具体的对抗性提示训练LLM通常在计算上是昂贵的，导致对连续松弛的依赖。与此同时，尽管具有有效性和泛化能力，但连续扰动训练并不总是能够捕获离散攻击所利用的全部漏洞。在这项工作中，我们的目标是通过引入MixAT来弥合这一差距，MixAT是一种新颖的方法，在训练期间结合了更强的离散攻击和更快的连续攻击。我们对MixAT进行了广泛的最先进攻击，提出了至少一次攻击成功率（ALO-ASB）指标来捕捉模型的最坏情况漏洞。我们表明，与之前的防御（ALO-ASB> 50%）相比，MixAT实现了更好的鲁棒性（ALO-ASB < 20%），同时保持与基于连续松弛的方法相当的运行时间。我们进一步分析了现实部署环境中的MixAT，探索聊天模板、量化、低等级适配器和温度如何影响对抗训练和评估，从而揭示了当前方法中的其他盲点。我们的结果表明，MixAT的离散-连续防御以最小的计算负担提供了原则性且卓越的鲁棒性-准确性权衡，凸显了其构建更安全的LLM的承诺。我们在https://github.com/insait-institute/MixAT上提供我们的代码和模型。



## **10. Untargeted Jailbreak Attack**

无目标越狱攻击 cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.02999v2) [paper-pdf](http://arxiv.org/pdf/2510.02999v2)

**Authors**: Xinzhe Huang, Wenjing Hu, Tianhang Zheng, Kedong Xiu, Xiaojun Jia, Di Wang, Zhan Qin, Kui Ren

**Abstract**: Existing gradient-based jailbreak attacks on Large Language Models (LLMs), such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimize adversarial suffixes to align the LLM output with a predefined target response. However, by restricting the optimization objective as inducing a predefined target, these methods inherently constrain the adversarial search space, which limit their overall attack efficacy. Furthermore, existing methods typically require a large number of optimization iterations to fulfill the large gap between the fixed target and the original model response, resulting in low attack efficiency.   To overcome the limitations of targeted jailbreak attacks, we propose the first gradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe response without enforcing any predefined patterns. Specifically, we formulate an untargeted attack objective to maximize the unsafety probability of the LLM response, which can be quantified using a judge model. Since the objective is non-differentiable, we further decompose it into two differentiable sub-objectives for optimizing an optimal harmful response and the corresponding adversarial prompt, with a theoretical analysis to validate the decomposition. In contrast to targeted jailbreak attacks, UJA's unrestricted objective significantly expands the search space, enabling a more flexible and efficient exploration of LLM vulnerabilities.Extensive evaluations demonstrate that UJA can achieve over 80% attack success rates against recent safety-aligned LLMs with only 100 optimization iterations, outperforming the state-of-the-art gradient-based attacks such as I-GCG and COLD-Attack by over 20%.

摘要: 现有的对大型语言模型（LLM）的基于梯度的越狱攻击，例如贪婪协调梯度（GCG）和COLD-Attack，通常会优化对抗性后缀，以将LLM输出与预定义的目标响应保持一致。然而，通过将优化目标限制为诱导预定义的目标，这些方法本质上限制了对抗搜索空间，从而限制了其总体攻击功效。此外，现有方法通常需要大量的优化迭代来满足固定目标和原始模型响应之间的大差距，导致攻击效率低。   为了克服定向越狱攻击的局限性，我们提出了第一个基于梯度的非定向越狱攻击（UJA），旨在在不强制执行任何预定义模式的情况下引发不安全的响应。具体来说，我们制定了一个无针对性的攻击目标，以最大化LLM响应的不安全概率，该概率可以使用判断模型进行量化。由于目标是不可微的，因此我们进一步将其分解为两个可微的子目标，用于优化最佳有害反应和相应的对抗提示，并通过理论分析来验证分解。与有针对性的越狱攻击相比，UJA的无限制目标显着扩大了搜索空间，从而能够更灵活、更高效地探索LLM漏洞。广泛的评估表明，UJA只需100次优化迭代即可针对最近的安全一致LLM实现超过80%的攻击成功率，比I-GCG和COLD-Attack等最先进的基于梯度的攻击性能高出20%以上。



## **11. Attention! Your Vision Language Model Could Be Maliciously Manipulated**

注意！您的视觉语言模型可能被恶意操纵 cs.CV

NeurIPS 2025

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2505.19911v2) [paper-pdf](http://arxiv.org/pdf/2505.19911v2)

**Authors**: Xiaosen Wang, Shaokang Wang, Zhijin Ge, Yuyang Luo, Shudong Zhang

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets. Code is available at https://github.com/Trustworthy-AI-Group/VMA.

摘要: 大型视觉语言模型（VLM）在理解复杂的现实世界场景和支持数据驱动的决策流程方面取得了显着的成功。然而，VLM对对抗性示例（无论是文本还是图像）表现出显着的脆弱性，这可能会导致各种对抗性结果，例如越狱、劫持和幻觉等。在这项工作中，我们从经验和理论上证明了VLM特别容易受到基于图像的对抗示例的影响，其中不可感知的扰动可以精确地操纵每个输出令牌。为此，我们提出了一种名为视觉语言模型操纵攻击（VMA）的新型攻击，该攻击将一阶和二阶动量优化技术与可微转换机制集成在一起，以有效地优化对抗性扰动。值得注意的是，VMA可以是一把双刃剑：它可以被用来实施各种攻击，例如越狱、劫持、隐私泄露、拒绝服务和海绵示例的生成等，同时允许注入水印以进行版权保护。广泛的实证评估证实了VMA在不同场景和数据集中的有效性和普遍性。代码可在https://github.com/Trustworthy-AI-Group/VMA上获取。



## **12. Fast-MIA: Efficient and Scalable Membership Inference for LLMs**

Fast-MIA：针对LLM的高效且可扩展的会员资格推断 cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.23074v1) [paper-pdf](http://arxiv.org/pdf/2510.23074v1)

**Authors**: Hiromu Takahashi, Shotaro Ishihara

**Abstract**: We propose Fast-MIA (https://github.com/Nikkei/fast-mia), a Python library for efficiently evaluating membership inference attacks (MIA) against Large Language Models (LLMs). MIA against LLMs has emerged as a crucial challenge due to growing concerns over copyright, security, and data privacy, and has attracted increasing research attention. However, the progress of this research is significantly hindered by two main obstacles: (1) the high computational cost of inference in LLMs, and (2) the lack of standardized and maintained implementations of MIA methods, which makes large-scale empirical comparison difficult. To address these challenges, our library provides fast batch inference and includes implementations of representative MIA methods under a unified evaluation framework. This library supports easy implementation of reproducible benchmarks with simple configuration and extensibility. We release Fast-MIA as an open-source (Apache License 2.0) tool to support scalable and transparent research on LLMs.

摘要: 我们提出Fast-MIA（https：//github.com/Nikkei/fast-mia），这是一个Python库，用于有效评估针对大型语言模型（LLM）的成员资格推理攻击（MIA）。由于对版权、安全和数据隐私的担忧日益加剧，针对LLM的MIA已成为一项关键挑战，并引起了越来越多的研究关注。然而，这项研究的进展受到两个主要障碍的严重阻碍：（1）LLM中推理的高计算成本，（2）MIA方法缺乏标准化和可维护的实现，这使得大规模的实证比较变得困难。为了应对这些挑战，我们的库提供快速批量推理，并包括统一评估框架下代表性MIA方法的实现。该库支持简单的配置和可扩展性轻松实现可重复的基准测试。我们将Fast-MIA作为开源（Apache许可证2.0）工具发布，以支持对LLM的可扩展和透明的研究。



## **13. MCPGuard : Automatically Detecting Vulnerabilities in MCP Servers**

MCPGuard：自动检测HCP服务器中的漏洞 cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.23673v1) [paper-pdf](http://arxiv.org/pdf/2510.23673v1)

**Authors**: Bin Wang, Zexin Liu, Hao Yu, Ao Yang, Yenan Huang, Jing Guo, Huangsheng Cheng, Hui Li, Huiyu Wu

**Abstract**: The Model Context Protocol (MCP) has emerged as a standardized interface enabling seamless integration between Large Language Models (LLMs) and external data sources and tools. While MCP significantly reduces development complexity and enhances agent capabilities, its openness and extensibility introduce critical security vulnerabilities that threaten system trustworthiness and user data protection. This paper systematically analyzes the security landscape of MCP-based systems, identifying three principal threat categories: (1) agent hijacking attacks stemming from protocol design deficiencies; (2) traditional web vulnerabilities in MCP servers; and (3) supply chain security. To address these challenges, we comprehensively survey existing defense strategies, examining both proactive server-side scanning approaches, ranging from layered detection pipelines and agentic auditing frameworks to zero-trust registry systems, and runtime interaction monitoring solutions that provide continuous oversight and policy enforcement. Our analysis reveals that MCP security fundamentally represents a paradigm shift where the attack surface extends from traditional code execution to semantic interpretation of natural language metadata, necessitating novel defense mechanisms tailored to this unique threat model.

摘要: 模型上下文协议（HCP）已成为一种标准化界面，能够实现大型语言模型（LLM）与外部数据源和工具之间的无缝集成。虽然LCP显着降低了开发复杂性并增强了代理能力，但其开放性和可扩展性引入了威胁系统可信度和用户数据保护的关键安全漏洞。本文系统地分析了基于MPP的系统的安全格局，确定了三种主要威胁类别：（1）源于协议设计缺陷的代理劫持攻击;（2）HCP服务器中的传统Web漏洞;（3）供应链安全。为了应对这些挑战，我们全面调查了现有的防御策略，研究了主动服务器端扫描方法（从分层检测管道和代理审计框架到零信任注册表系统）以及提供持续监督和政策执行的运行时交互监控解决方案。我们的分析表明，LCP安全从根本上代表了一种范式转变，攻击面从传统的代码执行扩展到自然语言元数据的语义解释，需要针对这种独特的威胁模型量身定制的新型防御机制。



## **14. Is Your Prompt Poisoning Code? Defect Induction Rates and Security Mitigation Strategies**

您的提示中毒代码是吗？缺陷诱导率和安全缓解策略 cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.22944v1) [paper-pdf](http://arxiv.org/pdf/2510.22944v1)

**Authors**: Bin Wang, YiLu Zhong, MiDi Wan, WenJie Yu, YuanBing Ouyang, Yenan Huang, Hui Li

**Abstract**: Large language models (LLMs) have become indispensable for automated code generation, yet the quality and security of their outputs remain a critical concern. Existing studies predominantly concentrate on adversarial attacks or inherent flaws within the models. However, a more prevalent yet underexplored issue concerns how the quality of a benign but poorly formulated prompt affects the security of the generated code. To investigate this, we first propose an evaluation framework for prompt quality encompassing three key dimensions: goal clarity, information completeness, and logical consistency. Based on this framework, we construct and publicly release CWE-BENCH-PYTHON, a large-scale benchmark dataset containing tasks with prompts categorized into four distinct levels of normativity (L0-L3). Extensive experiments on multiple state-of-the-art LLMs reveal a clear correlation: as prompt normativity decreases, the likelihood of generating insecure code consistently and markedly increases. Furthermore, we demonstrate that advanced prompting techniques, such as Chain-of-Thought and Self-Correction, effectively mitigate the security risks introduced by low-quality prompts, substantially improving code safety. Our findings highlight that enhancing the quality of user prompts constitutes a critical and effective strategy for strengthening the security of AI-generated code.

摘要: 大型语言模型（LLM）对于自动代码生成来说已不可或缺，但其输出的质量和安全性仍然是一个关键问题。现有的研究主要集中在对抗攻击或模型内的固有缺陷上。然而，一个更普遍但未充分研究的问题涉及良性但制定不当的提示的质量如何影响生成代码的安全性。为了研究这一点，我们首先提出了一个包含三个关键维度的即时质量评估框架：目标清晰度、信息完整性和逻辑一致性。基于此框架，我们构建并公开发布CWE-BENCH-PYTHON，这是一个大规模基准数据集，包含将提示分为四个不同的规范性级别（L0-L3）的任务。对多个最先进的LLM进行的广泛实验揭示了明显的相关性：随着即时规范性的降低，生成不安全代码的可能性持续且显着增加。此外，我们还证明，思想链和自我纠正等先进的提示技术可以有效地减轻低质量提示带来的安全风险，大大提高代码安全性。我们的研究结果强调，提高用户提示的质量是加强人工智能生成代码安全性的关键而有效的策略。



## **15. Sentra-Guard: A Multilingual Human-AI Framework for Real-Time Defense Against Adversarial LLM Jailbreaks**

Sentra-Guard：用于实时防御对抗LLM越狱的多语言人工智能框架 cs.CR

11 pages, 5 figures. Preprint version under review in the area of  Artificial Intelligence (cs.AI)

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22628v1) [paper-pdf](http://arxiv.org/pdf/2510.22628v1)

**Authors**: Md. Mehedi Hasan, Ziaur Rahman, Rafid Mostafiz, Md. Abir Hossain

**Abstract**: This paper presents a real-time modular defense system named Sentra-Guard. The system detects and mitigates jailbreak and prompt injection attacks targeting large language models (LLMs). The framework uses a hybrid architecture with FAISS-indexed SBERT embedding representations that capture the semantic meaning of prompts, combined with fine-tuned transformer classifiers, which are machine learning models specialized for distinguishing between benign and adversarial language inputs. It identifies adversarial prompts in both direct and obfuscated attack vectors. A core innovation is the classifier-retriever fusion module, which dynamically computes context-aware risk scores that estimate how likely a prompt is to be adversarial based on its content and context. The framework ensures multilingual resilience with a language-agnostic preprocessing layer. This component automatically translates non-English prompts into English for semantic evaluation, enabling consistent detection across over 100 languages. The system includes a HITL feedback loop, where decisions made by the automated system are reviewed by human experts for continual learning and rapid adaptation under adversarial pressure. Sentra-Guard maintains an evolving dual-labeled knowledge base of benign and malicious prompts, enhancing detection reliability and reducing false positives. Evaluation results show a 99.96% detection rate (AUC = 1.00, F1 = 1.00) and an attack success rate (ASR) of only 0.004%. This outperforms leading baselines such as LlamaGuard-2 (1.3%) and OpenAI Moderation (3.7%). Unlike black-box approaches, Sentra-Guard is transparent, fine-tunable, and compatible with diverse LLM backends. Its modular design supports scalable deployment in both commercial and open-source environments. The system establishes a new state-of-the-art in adversarial LLM defense.

摘要: 本文提出了一种实时模块化防御系统Sentra-Guard。该系统检测并缓解针对大型语言模型（LLM）的越狱和提示注入攻击。该框架使用混合架构，该架构具有FAISS索引的SBERT嵌入表示，该表示捕获提示的语义含义，并结合了微调的Transformer分类器，后者是专门用于区分良性和对抗性语言输入的机器学习模型。它可以识别直接攻击向量和混淆攻击向量中的对抗性提示。一个核心创新是分类器-检索器融合模块，该模块动态计算上下文感知风险评分，根据其内容和上下文估计提示具有对抗性的可能性。该框架通过语言不可知的预处理层确保多语言弹性。该组件自动将非英语提示翻译成英语进行语义评估，从而实现对100多种语言的一致检测。该系统包括一个HITL反馈循环，其中自动化系统做出的决策由人类专家审查，以便在对抗压力下持续学习和快速适应。Sentra-Guard维护不断发展的良性和恶意提示的双标签知识库，增强检测可靠性并减少误报。评估结果显示，检测率为99.96%（AUR = 1.00，F1 = 1.00），攻击成功率（ASB）仅为0.004%。这优于LlamaGuard-2（1.3%）和OpenAI Moderation（3.7%）等领先基准。与黑匣子方法不同，Sentra-Guard是透明的、可微调的，并与各种LLM后台兼容。其模块化设计支持商业和开源环境中的可扩展部署。该系统在对抗性LLM辩护方面建立了新的最新水平。



## **16. Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents**

打破代理主干：评估人工智能代理中主干LLM的安全性 cs.CR

Julia Bazinska and Max Mathys contributed equally

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22620v1) [paper-pdf](http://arxiv.org/pdf/2510.22620v1)

**Authors**: Julia Bazinska, Max Mathys, Francesco Casucci, Mateo Rojas-Carulla, Xander Davies, Alexandra Souly, Niklas Pfister

**Abstract**: AI agents powered by large language models (LLMs) are being deployed at scale, yet we lack a systematic understanding of how the choice of backbone LLM affects agent security. The non-deterministic sequential nature of AI agents complicates security modeling, while the integration of traditional software with AI components entangles novel LLM vulnerabilities with conventional security risks. Existing frameworks only partially address these challenges as they either capture specific vulnerabilities only or require modeling of complete agents. To address these limitations, we introduce threat snapshots: a framework that isolates specific states in an agent's execution flow where LLM vulnerabilities manifest, enabling the systematic identification and categorization of security risks that propagate from the LLM to the agent level. We apply this framework to construct the $\operatorname{b}^3$ benchmark, a security benchmark based on 194331 unique crowdsourced adversarial attacks. We then evaluate 31 popular LLMs with it, revealing, among other insights, that enhanced reasoning capabilities improve security, while model size does not correlate with security. We release our benchmark, dataset, and evaluation code to facilitate widespread adoption by LLM providers and practitioners, offering guidance for agent developers and incentivizing model developers to prioritize backbone security improvements.

摘要: 由大型语言模型（LLM）支持的人工智能代理正在大规模部署，但我们缺乏对主干LLM的选择如何影响代理安全性的系统了解。人工智能代理的非确定性顺序性质使安全建模变得复杂，而传统软件与人工智能组件的集成则使新型LLM漏洞与传统安全风险纠缠在一起。现有的框架只能部分解决这些挑战，因为它们要么仅捕获特定的漏洞，要么需要对完整的代理进行建模。为了解决这些限制，我们引入了威胁快照：一个框架，可以隔离代理执行流程中LLM漏洞表现的特定状态，从而能够系统地识别和分类从LLM传播到代理级别的安全风险。我们应用此框架来构建$\operatorName{b}'#39; 3 $基准，这是一个基于194331个独特众包对抗攻击的安全基准。然后，我们用它评估了31种流行的LLM，揭示了增强的推理能力可以提高安全性，而模型大小与安全性无关。我们发布我们的基准、数据集和评估代码，以促进LLM提供商和从业者的广泛采用，为代理开发人员提供指导，并激励模型开发人员优先考虑主干安全改进。



## **17. OFFSIDE: Benchmarking Unlearning Misinformation in Multimodal Large Language Models**

越位：多模式大型语言模型中消除错误信息的基准 cs.AI

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22535v1) [paper-pdf](http://arxiv.org/pdf/2510.22535v1)

**Authors**: Hao Zheng, Zirui Pang, Ling li, Zhijie Deng, Yuhan Pu, Zhaowei Zhu, Xiaobo Xia, Jiaheng Wei

**Abstract**: Advances in Multimodal Large Language Models (MLLMs) intensify concerns about data privacy, making Machine Unlearning (MU), the selective removal of learned information, a critical necessity. However, existing MU benchmarks for MLLMs are limited by a lack of image diversity, potential inaccuracies, and insufficient evaluation scenarios, which fail to capture the complexity of real-world applications. To facilitate the development of MLLMs unlearning and alleviate the aforementioned limitations, we introduce OFFSIDE, a novel benchmark for evaluating misinformation unlearning in MLLMs based on football transfer rumors. This manually curated dataset contains 15.68K records for 80 players, providing a comprehensive framework with four test sets to assess forgetting efficacy, generalization, utility, and robustness. OFFSIDE supports advanced settings like selective unlearning and corrective relearning, and crucially, unimodal unlearning (forgetting only text data). Our extensive evaluation of multiple baselines reveals key findings: (1) Unimodal methods (erasing text-based knowledge) fail on multimodal rumors; (2) Unlearning efficacy is largely driven by catastrophic forgetting; (3) All methods struggle with "visual rumors" (rumors appear in the image); (4) The unlearned rumors can be easily recovered and (5) All methods are vulnerable to prompt attacks. These results expose significant vulnerabilities in current approaches, highlighting the need for more robust multimodal unlearning solutions. The code is available at \href{https://github.com/zh121800/OFFSIDE}{https://github.com/zh121800/OFFSIDE}.

摘要: 多模式大型语言模型（MLLM）的进步加剧了人们对数据隐私的担忧，使机器取消学习（MU）（选择性地删除学习信息）成为一种至关重要的必要性。然而，MLLM的现有MU基准受到缺乏图像多样性、潜在不准确性和评估场景不足的限制，无法捕捉现实世界应用程序的复杂性。为了促进MLLMs的发展，消除学习和减轻上述限制，我们引入OFFSIDE，一个新的基准评估错误信息的学习在MLLMs的基础上足球转会谣言。这个手动策划的数据集包含80个玩家的15.68K记录，提供了一个全面的框架，其中包括四个测试集，以评估遗忘效率，泛化，实用性和鲁棒性。OFFSIDE支持高级设置，如选择性遗忘和纠正性重新学习，以及至关重要的单峰遗忘（只忘记文本数据）。我们对多个基线的广泛评估揭示了关键发现：（1）单峰方法（擦除基于文本的知识）在多模态谣言上失败;（2）遗忘功效在很大程度上是由灾难性遗忘驱动的;（3）所有方法都与“视觉谣言”（谣言出现在图像中）作斗争;（4）未学习的谣言可以很容易地恢复;（5）所有方法都容易受到即时攻击。这些结果暴露了当前方法中的显着漏洞，凸显了对更强大的多模式去学习解决方案的需求。该代码可访问\href{https：//github.com/zh121800/OFSIDE}{https：//github.com/zh121800/OFSIDE}。



## **18. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

令人沮丧的简单但高效的攻击基线：针对GPT-4.5/4 o/o 1的强黑匣子模型的成功率超过90% cs.CV

NeurIPS 2025. Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2503.10635v2) [paper-pdf](http://arxiv.org/pdf/2503.10635v2)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against closed-source commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial black-box LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we propose to refine semantic clarity by encoding explicit semantic details within local regions, thus ensuring the capture of finer-grained features and inter-model transferability, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective baseline: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. While the naive source-target matching method has been utilized before in the literature, we are the first to provide a tight analysis, which establishes a close connection between perturbation optimization and semantics. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5/3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods with lower $\ell_1/\ell_2$ perturbations.

摘要: 尽管在开源大型视觉语言模型（LVLM）上表现出色，但基于传输的有针对性的攻击往往无法对封闭源商业LVLM进行攻击。分析失败的对抗性扰动表明，习得的扰动通常源于均匀分布，并且缺乏明确的语义细节，从而导致意外反应。语义信息的严重缺失导致商业黑匣子LVLM要么完全忽略扰动，要么误解其嵌入式语义，从而导致攻击失败。为了克服这些问题，我们建议通过在局部区域内编码显式的语义细节来完善语义清晰度，从而确保捕获更细粒度的特征和模型间的可移植性，并通过将修改集中在语义丰富的区域而不是统一应用它们。为了实现这一目标，我们提出了一个简单但高效的基线：在每个优化步骤中，通过受控的长宽比和比例随机裁剪对抗图像，调整大小，然后与嵌入空间中的目标图像对齐。虽然文献中以前曾使用过朴素的源目标匹配方法，但我们是第一个提供严密分析的人，该分析在扰动优化和语义之间建立了密切的联系。实验结果证实了我们的假设。我们用专注于关键区域的局部聚集扰动制作的对抗性示例表现出了令人惊讶的良好可移植性，包括GPT-4.5、GPT-4 o、Gemini-2.0-Flash、Claude-3.5/3.7-十四行诗，甚至还有像o 1、Claude-3.7-思考和Gemini-2.0-闪光思考这样的推理模型。我们的方法在GPT-4.5，4 o和o 1上实现了超过90%的成功率，显著优于所有现有的最先进的攻击方法，具有更低的$\ell_1/\ell_2$扰动。



## **19. Jailbreak Mimicry: Automated Discovery of Narrative-Based Jailbreaks for Large Language Models**

越狱模仿：大型语言模型基于叙事的越狱的自动发现 cs.CR

18 pages, 5 figures

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.22085v1) [paper-pdf](http://arxiv.org/pdf/2510.22085v1)

**Authors**: Pavlos Ntais

**Abstract**: Large language models (LLMs) remain vulnerable to sophisticated prompt engineering attacks that exploit contextual framing to bypass safety mechanisms, posing significant risks in cybersecurity applications. We introduce Jailbreak Mimicry, a systematic methodology for training compact attacker models to automatically generate narrative-based jailbreak prompts in a one-shot manner. Our approach transforms adversarial prompt discovery from manual craftsmanship into a reproducible scientific process, enabling proactive vulnerability assessment in AI-driven security systems. Developed for the OpenAI GPT-OSS-20B Red-Teaming Challenge, we use parameter-efficient fine-tuning (LoRA) on Mistral-7B with a curated dataset derived from AdvBench, achieving an 81.0% Attack Success Rate (ASR) against GPT-OSS-20B on a held-out test set of 200 items. Cross-model evaluation reveals significant variation in vulnerability patterns: our attacks achieve 66.5% ASR against GPT-4, 79.5% on Llama-3 and 33.0% against Gemini 2.5 Flash, demonstrating both broad applicability and model-specific defensive strengths in cybersecurity contexts. This represents a 54x improvement over direct prompting (1.5% ASR) and demonstrates systematic vulnerabilities in current safety alignment approaches. Our analysis reveals that technical domains (Cybersecurity: 93% ASR) and deception-based attacks (Fraud: 87.8% ASR) are particularly vulnerable, highlighting threats to AI-integrated threat detection, malware analysis, and secure systems, while physical harm categories show greater resistance (55.6% ASR). We employ automated harmfulness evaluation using Claude Sonnet 4, cross-validated with human expert assessment, ensuring reliable and scalable evaluation for cybersecurity red-teaming. Finally, we analyze failure mechanisms and discuss defensive strategies to mitigate these vulnerabilities in AI for cybersecurity.

摘要: 大型语言模型（LLM）仍然容易受到复杂的即时工程攻击，这些攻击利用上下文框架来绕过安全机制，从而给网络安全应用带来重大风险。我们引入了越狱模仿，这是一种系统性方法，用于训练紧凑的攻击者模型，以一次性方式自动生成基于叙述的越狱提示。我们的方法将对抗性即时发现从手工工艺转变为可重复的科学过程，从而在人工智能驱动的安全系统中实现主动的漏洞评估。我们为OpenAI GPT-OSS-20 B Red-Teaming Challenge而开发，在Mistral-7 B上使用参数高效微调（LoRA），采用源自AdvBench的精心策划数据集，在包含200个项目的测试集上实现了针对GPT-OSS-20 B的81.0%的攻击成功率（ASB）。跨模型评估揭示了漏洞模式的显著差异：我们的攻击对GPT-4实现了66.5%的ASR，对Llama-3实现了79.5%的ASR，对Gemini 2.5 Flash实现了33.0%的ASR，这表明了网络安全环境中的广泛适用性和特定于模型的防御优势。这比直接提示（1.5% ASR）提高了54倍，并表明当前安全调整方法存在系统性漏洞。我们的分析显示，技术领域（网络安全：93% ASR）和基于欺骗的攻击（欺诈：87.8% ASR）特别脆弱，突出了对AI集成威胁检测，恶意软件分析和安全系统的威胁，而物理伤害类别显示出更大的抵抗力（55.6% ASR）。我们使用Claude Sonnet 4进行自动危害性评估，并与人类专家评估进行交叉验证，确保对网络安全红队进行可靠和可扩展的评估。最后，我们分析了故障机制并讨论了缓解人工智能网络安全中这些漏洞的防御策略。



## **20. Toward Understanding the Transferability of Adversarial Suffixes in Large Language Models**

了解大型语言模型中对抗性后缀的可移植性 cs.CL

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.22014v1) [paper-pdf](http://arxiv.org/pdf/2510.22014v1)

**Authors**: Sarah Ball, Niki Hasrati, Alexander Robey, Avi Schwarzschild, Frauke Kreuter, Zico Kolter, Andrej Risteski

**Abstract**: Discrete optimization-based jailbreaking attacks on large language models aim to generate short, nonsensical suffixes that, when appended onto input prompts, elicit disallowed content. Notably, these suffixes are often transferable -- succeeding on prompts and models for which they were never optimized. And yet, despite the fact that transferability is surprising and empirically well-established, the field lacks a rigorous analysis of when and why transfer occurs. To fill this gap, we identify three statistical properties that strongly correlate with transfer success across numerous experimental settings: (1) how much a prompt without a suffix activates a model's internal refusal direction, (2) how strongly a suffix induces a push away from this direction, and (3) how large these shifts are in directions orthogonal to refusal. On the other hand, we find that prompt semantic similarity only weakly correlates with transfer success. These findings lead to a more fine-grained understanding of transferability, which we use in interventional experiments to showcase how our statistical analysis can translate into practical improvements in attack success.

摘要: 对大型语言模型的基于离散优化的越狱攻击旨在生成简短、无意义的后缀，当附加到输入提示时，会引发不允许的内容。值得注意的是，这些后缀通常是可移植的--在从未对其进行过优化的提示和模型上取得成功。然而，尽管可转移性令人惊讶并且在经验上得到了充分的证实，但该领域缺乏对何时和为何发生转移的严格分析。为了填补这一空白，我们确定了三个与众多实验环境中的转移成功密切相关的统计属性：（1）没有后缀的提示在多大程度上激活了模型的内部拒绝方向，（2）后缀引发了如何强烈的推动远离这个方向，以及（3）这些变化在与拒绝垂直的方向上有多大。另一方面，我们发现提示的语义相似性与转移成功仅弱相关。这些发现使我们对可移植性有了更细的理解，我们在干预实验中使用它来展示我们的统计分析如何转化为攻击成功的实际改进。



## **21. Memory Injection Attacks on LLM Agents via Query-Only Interaction**

通过仅查询交互对LLM代理进行内存注入攻击 cs.LG

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2503.03704v3) [paper-pdf](http://arxiv.org/pdf/2503.03704v3)

**Authors**: Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, Zhen Xiang

**Abstract**: Agents powered by large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, without assuming that the attacker can directly modify the memory bank of the agent. The attacker injects malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps corresponding to a different target query during the agent's execution of the victim user's query. Specifically, we introduce a sequence of bridging steps to link victim queries to the malicious reasoning steps. During the memory injection, we propose an indication prompt that guides the agent to autonomously generate similar bridging steps, with a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing later victim queries. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting the risk.

摘要: 由大型语言模型（LLM）支持的代理在各种复杂的现实世界应用程序中表现出了强大的能力。然而，当为演示而检索的过去记录是恶意的时，内存库受损的LLM代理可能很容易产生有害输出。在本文中，我们提出了一种新型的内存注入攻击MINJA，但没有假设攻击者可以直接修改代理的内存库。攻击者仅通过查询和输出观察与代理交互，将恶意记录注入内存库。这些恶意记录被设计为在代理执行受害者用户的查询期间引出对应于不同目标查询的恶意推理步骤序列。具体来说，我们引入了一系列的桥接步骤，将受害者查询链接到恶意推理步骤。在内存注入过程中，我们提出了一个指示提示，引导代理自主生成类似的桥接步骤，逐步缩短策略，逐渐删除指示提示，这样的恶意记录将很容易被检索时，处理以后的受害者查询。我们在不同的代理广泛的实验证明了MINJA在损害代理内存的有效性。MINJA对执行的要求最低，使任何用户都能影响代理内存，从而凸显风险。



## **22. Uncovering the Persuasive Fingerprint of LLMs in Jailbreaking Attacks**

在越狱袭击中发现LLM的说服指纹 cs.CL

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21983v1) [paper-pdf](http://arxiv.org/pdf/2510.21983v1)

**Authors**: Havva Alizadeh Noughabi, Julien Serbanescu, Fattane Zarrinkalam, Ali Dehghantanha

**Abstract**: Despite recent advances, Large Language Models remain vulnerable to jailbreak attacks that bypass alignment safeguards and elicit harmful outputs. While prior research has proposed various attack strategies differing in human readability and transferability, little attention has been paid to the linguistic and psychological mechanisms that may influence a model's susceptibility to such attacks. In this paper, we examine an interdisciplinary line of research that leverages foundational theories of persuasion from the social sciences to craft adversarial prompts capable of circumventing alignment constraints in LLMs. Drawing on well-established persuasive strategies, we hypothesize that LLMs, having been trained on large-scale human-generated text, may respond more compliantly to prompts with persuasive structures. Furthermore, we investigate whether LLMs themselves exhibit distinct persuasive fingerprints that emerge in their jailbreak responses. Empirical evaluations across multiple aligned LLMs reveal that persuasion-aware prompts significantly bypass safeguards, demonstrating their potential to induce jailbreak behaviors. This work underscores the importance of cross-disciplinary insight in addressing the evolving challenges of LLM safety. The code and data are available.

摘要: 尽管最近取得了进步，但大型语言模型仍然容易受到越狱攻击，这些攻击绕过对齐保障措施并引发有害输出。虽然之前的研究提出了各种在人类可读性和可移植性方面不同的攻击策略，但很少有人关注可能影响模型对此类攻击的易感性的语言和心理机制。在本文中，我们研究了一项跨学科的研究，该研究利用社会科学的说服基础理论来设计能够规避法学硕士中的一致限制的对抗性提示。利用成熟的说服策略，我们假设LLM在接受过大规模人类生成的文本训练后，可能会对具有说服力结构的提示做出更顺从的反应。此外，我们还调查LLM本身是否在越狱反应中表现出明显的有说服力的指纹。对多个一致的LLM进行的经验评估显示，说服意识的提示大大绕过了保障措施，证明了它们有可能引发越狱行为。这项工作强调了跨学科洞察力在应对LLM安全不断变化的挑战方面的重要性。代码和数据均可用。



## **23. $δ$-STEAL: LLM Stealing Attack with Local Differential Privacy**

$δ$-STEAL：具有本地差异隐私的LLM窃取攻击 cs.CR

Accepted at ACML 2025 (PMLR W&CP). Code:  https://github.com/kirudang/LDP_Stealing_Attack

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21946v1) [paper-pdf](http://arxiv.org/pdf/2510.21946v1)

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah

**Abstract**: Large language models (LLMs) demonstrate remarkable capabilities across various tasks. However, their deployment introduces significant risks related to intellectual property. In this context, we focus on model stealing attacks, where adversaries replicate the behaviors of these models to steal services. These attacks are highly relevant to proprietary LLMs and pose serious threats to revenue and financial stability. To mitigate these risks, the watermarking solution embeds imperceptible patterns in LLM outputs, enabling model traceability and intellectual property verification. In this paper, we study the vulnerability of LLM service providers by introducing $\delta$-STEAL, a novel model stealing attack that bypasses the service provider's watermark detectors while preserving the adversary's model utility. $\delta$-STEAL injects noise into the token embeddings of the adversary's model during fine-tuning in a way that satisfies local differential privacy (LDP) guarantees. The adversary queries the service provider's model to collect outputs and form input-output training pairs. By applying LDP-preserving noise to these pairs, $\delta$-STEAL obfuscates watermark signals, making it difficult for the service provider to determine whether its outputs were used, thereby preventing claims of model theft. Our experiments show that $\delta$-STEAL with lightweight modifications achieves attack success rates of up to $96.95\%$ without significantly compromising the adversary's model utility. The noise scale in LDP controls the trade-off between attack effectiveness and model utility. This poses a significant risk, as even robust watermarks can be bypassed, allowing adversaries to deceive watermark detectors and undermine current intellectual property protection methods.

摘要: 大型语言模型（LLM）在各种任务中展示了非凡的能力。然而，它们的部署会带来与知识产权相关的重大风险。在此背景下，我们重点关注模型窃取攻击，其中对手复制这些模型的行为来窃取服务。这些攻击与专有LLM高度相关，并对收入和财务稳定构成严重威胁。为了降低这些风险，水印解决方案在LLM输出中嵌入不可感知的模式，从而实现模型可追溯性和知识产权验证。本文通过引入$\delta$-STEAL来研究LLM服务提供商的漏洞，这是一种新型模型窃取攻击，可以绕过服务提供商的水印检测器，同时保留对手的模型效用。$\delta$-STEAL在微调期间以满足本地差异隐私（SDP）保证的方式将噪音注入对手模型的令牌嵌入中。对手查询服务提供商的模型以收集输出并形成输入输出训练对。通过对这些对应用LP保留噪音，$\delta$-STEAL会混淆水印信号，使服务提供商难以确定其输出是否被使用，从而防止模型盗窃的指控。我们的实验表明，经过轻量级修改的$\delta$-STEAL的攻击成功率高达96.95美元\%$，而不会显着损害对手的模型效用。SDP中的噪音规模控制着攻击有效性和模型效用之间的权衡。这构成了巨大的风险，因为即使是强大的水印也可能被绕过，从而使对手欺骗水印检测器并破坏当前的知识产权保护方法。



## **24. Adversarial Déjà Vu: Jailbreak Dictionary Learning for Stronger Generalization to Unseen Attacks**

对抗性Déjà Vu：越狱词典学习以更强有力地概括隐形攻击 cs.LG

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21910v1) [paper-pdf](http://arxiv.org/pdf/2510.21910v1)

**Authors**: Mahavir Dabas, Tran Huynh, Nikhil Reddy Billa, Jiachen T. Wang, Peng Gao, Charith Peris, Yao Ma, Rahul Gupta, Ming Jin, Prateek Mittal, Ruoxi Jia

**Abstract**: Large language models remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Defending against novel jailbreaks represents a critical challenge in AI safety. Adversarial training -- designed to make models robust against worst-case perturbations -- has been the dominant paradigm for adversarial robustness. However, due to optimization challenges and difficulties in defining realistic threat models, adversarial training methods often fail on newly developed jailbreaks in practice. This paper proposes a new paradigm for improving robustness against unseen jailbreaks, centered on the Adversarial D\'ej\`a Vu hypothesis: novel jailbreaks are not fundamentally new, but largely recombinations of adversarial skills from previous attacks. We study this hypothesis through a large-scale analysis of 32 attack papers published over two years. Using an automated pipeline, we extract and compress adversarial skills into a sparse dictionary of primitives, with LLMs generating human-readable descriptions. Our analysis reveals that unseen attacks can be effectively explained as sparse compositions of earlier skills, with explanatory power increasing monotonically as skill coverage grows. Guided by this insight, we introduce Adversarial Skill Compositional Training (ASCoT), which trains on diverse compositions of skill primitives rather than isolated attack instances. ASCoT substantially improves robustness to unseen attacks, including multi-turn jailbreaks, while maintaining low over-refusal rates. We also demonstrate that expanding adversarial skill coverage, not just data scale, is key to defending against novel attacks. \textcolor{red}{\textbf{Warning: This paper contains content that may be harmful or offensive in nature.

摘要: 大型语言模型仍然容易受到越狱攻击，这些攻击绕过安全护栏，引发有害输出。防御新型越狱是人工智能安全的一个关键挑战。对抗性训练--旨在使模型在最坏情况下保持稳健性--一直是对抗性稳健性的主要范式。然而，由于优化挑战和定义现实威胁模型的困难，对抗性训练方法在实践中常常在新开发的越狱中失败。本文以对抗D ' ej ' a Vu假设为中心，提出了一种用于提高针对未见越狱的鲁棒性的新范式：新型越狱从根本上来说并不是新的，而是之前攻击中对抗技能的重新组合。我们通过对两年内发表的32篇攻击论文的大规模分析来研究这一假设。使用自动化管道，我们将对抗技能提取并压缩到稀疏的基元字典中，由LLM生成人类可读的描述。我们的分析表明，不可见的攻击可以有效地解释为早期技能的稀疏组成，并且随着技能覆盖范围的增加，解释能力单调增加。在这一见解的指导下，我们引入了对抗性技能合成训练（ASCoT），该训练基于技能基元的不同合成，而不是孤立的攻击实例。ASCoT大幅提高了对不可见攻击（包括多回合越狱）的稳健性，同时保持较低的过度拒绝率。我们还证明，扩大对抗技能覆盖范围，而不仅仅是数据规模，是防御新型攻击的关键。\textColor{red}{\textBF{警告：本文包含可能有害或冒犯性的内容。



## **25. Detecting Various DeFi Price Manipulations with LLM Reasoning**

使用LLM推理检测各种DeFi价格操纵 cs.CR

Accepted by ASE 2025. Please cite the conference version of this  paper, e.g., "Juantao Zhong, Daoyuan Wu, Ye Liu, Maoyi Xie, Yang Liu, Yi Li,  Ning Liu. Detecting Various DeFi Price Manipulations with LLM Reasoning. In  40th IEEE/ACM International Conference on Automated Software Engineering (ASE  2025)"

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2502.11521v2) [paper-pdf](http://arxiv.org/pdf/2502.11521v2)

**Authors**: Juantao Zhong, Daoyuan Wu, Ye Liu, Maoyi Xie, Yang Liu, Yi Li, Ning Liu

**Abstract**: DeFi (Decentralized Finance) is one of the most important applications of today's cryptocurrencies and smart contracts. It manages hundreds of billions in Total Value Locked (TVL) on-chain, yet it remains susceptible to common DeFi price manipulation attacks. Despite state-of-the-art (SOTA) systems like DeFiRanger and DeFort, we found that they are less effective to non-standard price models in custom DeFi protocols, which account for 44.2% of the 95 DeFi price manipulation attacks reported over the past three years. In this paper, we introduce the first LLM-based approach, DeFiScope, for detecting DeFi price manipulation attacks in both standard and custom price models. Our insight is that large language models (LLMs) have certain intelligence to abstract price calculation from smart contract source code and infer the trend of token price changes based on the extracted price models. To further strengthen LLMs in this aspect, we leverage Foundry to synthesize on-chain data and use it to fine-tune a DeFi price-specific LLM. Together with the high-level DeFi operations recovered from low-level transaction data, DeFiScope detects various DeFi price manipulations according to systematically mined patterns. Experimental results show that DeFiScope achieves a high recall of 80% on real-world attacks, a precision of 96% on suspicious transactions, and zero false alarms on benign transactions, significantly outperforming SOTA approaches. Moreover, we evaluate DeFiScope's cost-effectiveness and demonstrate its practicality by helping our industry partner confirm 147 real-world price manipulation attacks, including discovering 81 previously unknown historical incidents.

摘要: DeFi（去中心化金融）是当今加密货币和智能合约最重要的应用之一。它管理着数千亿美元的链上总价值锁定（TFL），但仍然容易受到常见的DeFi价格操纵攻击。尽管有DeFiRanger和DeFort等最先进的（SOTA）系统，但我们发现它们对自定义DeFi协议中的非标准价格模型效果较差，该协议占过去三年报告的95起DeFi价格操纵攻击的44.2%。在本文中，我们介绍了第一种基于LLM的方法DeFiScope，用于检测标准和定制价格模型中的DeFi价格操纵攻击。我们的见解是，大型语言模型（LLM）具有一定的智能，可以从智能合约源代码中提取价格计算，并根据提取的价格模型推断代币价格变化的趋势。为了进一步加强这方面的LLM，我们利用Foundry来合成链上数据，并使用它来微调DeFi特定价格的LLM。DeFiScope与从低级交易数据恢复的高级DeFi操作一起，根据系统挖掘的模式检测各种DeFi价格操纵。实验结果表明，DeFiScope在现实世界攻击中实现了80%的高召回率，在可疑交易中实现了96%的准确率，在良性交易中实现了零误报，显着优于SOTA方法。此外，我们评估DeFiScope的成本效益，并通过帮助我们的行业合作伙伴确认147次现实世界的价格操纵攻击（包括发现81起之前未知的历史事件）来证明其实用性。



## **26. SBASH: a Framework for Designing and Evaluating RAG vs. Prompt-Tuned LLM Honeypots**

SBASH：设计和评估RAG与预算调整LLM蜜罐的框架 cs.CR

to be published in: The 3rd International Conference on Foundation  and Large Language Models (FLLM2025), IEEE, 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21459v1) [paper-pdf](http://arxiv.org/pdf/2510.21459v1)

**Authors**: Adetayo Adebimpe, Helmut Neukirchen, Thomas Welsh

**Abstract**: Honeypots are decoy systems used for gathering valuable threat intelligence or diverting attackers away from production systems. Maximising attacker engagement is essential to their utility. However research has highlighted that context-awareness, such as the ability to respond to new attack types, systems and attacker agents, is necessary to increase engagement. Large Language Models (LLMs) have been shown as one approach to increase context awareness but suffer from several challenges including accuracy and timeliness of response time, high operational costs and data-protection issues due to cloud deployment. We propose the System-Based Attention Shell Honeypot (SBASH) framework which manages data-protection issues through the use of lightweight local LLMs. We investigate the use of Retrieval Augmented Generation (RAG) supported LLMs and non-RAG LLMs for Linux shell commands and evaluate them using several different metrics such as response time differences, realism from human testers, and similarity to a real system calculated with Levenshtein distance, SBert, and BertScore. We show that RAG improves accuracy for untuned models while models that have been tuned via a system prompt that tells the LLM to respond like a Linux system achieve without RAG a similar accuracy as untuned with RAG, while having a slightly lower latency.

摘要: 蜜罐是诱饵系统，用于收集有价值的威胁情报或将攻击者从生产系统转移出去。最大限度地提高攻击者的参与度对于它们的实用性至关重要。然而，研究强调，上下文感知（例如响应新攻击类型、系统和攻击者代理的能力）对于提高参与度是必要的。大型语言模型（LLM）已被证明是提高上下文感知能力的一种方法，但也面临着诸多挑战，包括响应时间的准确性和及时性、高运营成本以及云部署带来的数据保护问题。我们提出了基于系统的注意力Shell蜜罐（SBASH）框架，该框架通过使用轻量级本地LLM来管理数据保护问题。我们调查了对Linux shell命令使用检索增强生成（RAG）支持的LLM和非RAG LLM，并使用几种不同的指标来评估它们，例如响应时间差异、人类测试人员的真实性以及与使用Levenshtein距离、SBert和BertScore计算的真实系统的相似性。我们表明，RAG提高了未调优模型的准确性，而通过系统提示（告诉LLM像Linux系统一样响应）进行调优的模型在没有RAG的情况下可以实现与未调优RAG类似的准确性，同时具有稍低的延迟。



## **27. FLAMES: Fine-tuning LLMs to Synthesize Invariants for Smart Contract Security**

FLAMES：微调LLM以合成不变量以实现智能合同安全 cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21401v1) [paper-pdf](http://arxiv.org/pdf/2510.21401v1)

**Authors**: Mojtaba Eshghie, Gabriele Morello, Matteo Lauretano, Alexandre Bartel, Martin Monperrus

**Abstract**: Smart contract vulnerabilities cost billions of dollars annually, yet existing automated analysis tools fail to generate deployable defenses. We present FLAMES, a novel automated approach that synthesizes executable runtime guards as Solidity "require" statements to harden smart contracts against exploits. Unlike prior work that relies on vulnerability labels, symbolic analysis, or natural language specifications, FLAMES employs domain-adapted large language models trained through fill-in-the-middle supervised fine-tuning on real-world invariants extracted from 514,506 verified contracts. Our extensive evaluation across three dimensions demonstrates FLAMES's effectiveness: (1) Compilation: FLAMES achieves 96.7% compilability for synthesized invariant (2) Semantic Quality: on a curated test set of 5,000 challenging invariants, FLAMES produces exact or semantically equivalent matches to ground truth in 44.5% of cases; (3) Exploit Mitigation: FLAMES prevents 22 out of 108 real exploits (20.4%) while preserving contract functionality, and (4) FLAMES successfully blocks the real-world APEMAGA incident by synthesizing a pre-condition that mitigates the attack. FLAMES establishes that domain-adapted LLMs can automatically generate production-ready security defenses for smart contracts without requiring vulnerability detection, formal specifications, or human intervention. We release our code, model weights, datasets, and evaluation infrastructure to enable reproducible research in this critical domain.

摘要: 智能合同漏洞每年损失数十亿美元，但现有的自动化分析工具无法生成可部署的防御。我们介绍了FLAMES，这是一种新颖的自动化方法，它将可执行运行时防护合成为Solidity“要求”声明，以强化智能合同以防止漏洞利用。与之前依赖于漏洞标签、符号分析或自然语言规范的工作不同，FLAMES采用了自适应域的大型语言模型，该模型通过对从514，506个已验证合同中提取的现实世界不变量进行中间填充监督微调来训练。我们在三个维度上的广泛评估证明了FLAMES的有效性：（1）编译：FLAMES对于合成不变量实现了96.7%的可编译性（2）语义质量：在由5，000个具有挑战性的不变量组成的精心策划的测试集上，FLAMES在44.5%的情况下生成与基本事实的精确或语义等效的匹配;（3）利用缓解：FLAMES阻止了108个真实漏洞中的22个（20.4%），同时保留了合同功能，（4）FLAMES通过合成减轻攻击的先决条件，成功阻止了现实世界的APEMAGA事件。FLAMES确立，自适应域的LLM可以自动为智能合同生成可生产的安全防御，而无需漏洞检测、正式规范或人为干预。我们发布我们的代码、模型权重、数据集和评估基础设施，以实现这一关键领域的可重复研究。



## **28. Reverse Engineering Human Preferences with Reinforcement Learning**

利用强化学习反向工程人类偏好 cs.CL

NeurIPS 2025 (Spotlight)

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2505.15795v2) [paper-pdf](http://arxiv.org/pdf/2505.15795v2)

**Authors**: Lisa Alazraki, Tan Yi-Chern, Jon Ander Campos, Maximilian Mozes, Marek Rei, Max Bartolo

**Abstract**: The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.

摘要: 大型语言模型（LLM）的能力通常由其他经过训练以预测人类偏好的LLM进行评估。这个框架（称为LLM as-a-Judge）具有高度可扩展性且成本相对较低。然而，它也容易受到恶意利用，因为LLM响应可以被调整以过度适应法官的偏好。之前的工作表明，候选人LLM生成的答案可以事后编辑，以最大限度地提高法官LLM分配给它们的分数。在这项研究中，我们采用了一种不同的方法，并使用judge-LLM提供的信号作为奖励，以对抗性地调整模型，这些模型生成旨在提高下游性能的文本前置码。我们发现，使用这些模型流水线化的冻结LLM比现有框架获得更高的LLM评估分数。至关重要的是，与直接干预模型响应的其他框架不同，我们的方法几乎无法检测。我们还证明，当候选LLM和判断LLM被训练期间未使用的模型替换时，调整后的前同步码生成器的有效性会转移。这些发现提出了有关设计更可靠的法学硕士作为法官评估环境的重要问题。他们还证明，人类偏好可以通过管道化LLM来通过强化学习优化上游前级，从而有效地反向设计--这种方法可以在对抗性攻击之外的各种任务和领域中找到未来的应用。



## **29. LLM-Powered Detection of Price Manipulation in DeFi**

LLM支持的DeFi价格操纵检测 cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21272v1) [paper-pdf](http://arxiv.org/pdf/2510.21272v1)

**Authors**: Lu Liu, Wuqi Zhang, Lili Wei, Hao Guan, Yongqiang Tian, Yepang Liu

**Abstract**: Decentralized Finance (DeFi) smart contracts manage billions of dollars, making them a prime target for exploits. Price manipulation vulnerabilities, often via flash loans, are a devastating class of attacks causing significant financial losses. Existing detection methods are limited. Reactive approaches analyze attacks only after they occur, while proactive static analysis tools rely on rigid, predefined heuristics, limiting adaptability. Both depend on known attack patterns, failing to identify novel variants or comprehend complex economic logic. We propose PMDetector, a hybrid framework combining static analysis with Large Language Model (LLM)-based reasoning to proactively detect price manipulation vulnerabilities. Our approach uses a formal attack model and a three-stage pipeline. First, static taint analysis identifies potentially vulnerable code paths. Second, a two-stage LLM process filters paths by analyzing defenses and then simulates attacks to evaluate exploitability. Finally, a static analysis checker validates LLM results, retaining only high-risk paths and generating comprehensive vulnerability reports. To evaluate its effectiveness, we built a dataset of 73 real-world vulnerable and 288 benign DeFi protocols. Results show PMDetector achieves 88% precision and 90% recall with Gemini 2.5-flash, significantly outperforming state-of-the-art static analysis and LLM-based approaches. Auditing a vulnerability with PMDetector costs just $0.03 and takes 4.0 seconds with GPT-4.1, offering an efficient and cost-effective alternative to manual audits.

摘要: 去中心化金融（DeFi）智能合同管理着数十亿美元，使其成为漏洞利用的主要目标。价格操纵漏洞（通常通过闪电贷款）是一类毁灭性的攻击，会造成重大财务损失。现有的检测方法有限。反应式方法仅在攻击发生后对其进行分析，而主动静态分析工具依赖于严格的、预定义的启发式方法，限制了适应性。两者都依赖于已知的攻击模式，无法识别新颖的变体或理解复杂的经济逻辑。我们提出PMDDetector，这是一个将静态分析与基于大型语言模型（LLM）的推理相结合的混合框架，可以主动检测价格操纵漏洞。我们的方法使用正式攻击模型和三阶段管道。首先，静态污染分析识别潜在脆弱的代码路径。其次，两阶段LLM流程通过分析防御来过滤路径，然后模拟攻击以评估可利用性。最后，静态分析检查器验证LLM结果，仅保留高风险路径并生成全面的漏洞报告。为了评估其有效性，我们构建了一个包含73个现实世界脆弱协议和288个良性DeFi协议的数据集。结果显示，PMDetector使用Gemini 2.5-Flash实现了88%的准确率和90%的召回率，显着优于最先进的静态分析和基于LLM的方法。使用PMDetector审计漏洞只需0.03美元，使用GPT-4.1只需4.0秒，为手动审计提供了高效且经济实惠的替代方案。



## **30. Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data**

LLM上的病毒感染攻击：您的中毒可以“通过”合成数据传播 cs.CR

Camera Ready of NeurIPS 2025 Spotlight. Source code:  https://github.com/liangzid/VirusInfectionAttack

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2509.23041v2) [paper-pdf](http://arxiv.org/pdf/2509.23041v2)

**Authors**: Zi Liang, Qingqing Ye, Xuan Liu, Yanyun Wang, Jianliang Xu, Haibo Hu

**Abstract**: Synthetic data refers to artificial samples generated by models. While it has been validated to significantly enhance the performance of large language models (LLMs) during training and has been widely adopted in LLM development, potential security risks it may introduce remain uninvestigated. This paper systematically evaluates the resilience of synthetic-data-integrated training paradigm for LLMs against mainstream poisoning and backdoor attacks. We reveal that such a paradigm exhibits strong resistance to existing attacks, primarily thanks to the different distribution patterns between poisoning data and queries used to generate synthetic samples. To enhance the effectiveness of these attacks and further investigate the security risks introduced by synthetic data, we introduce a novel and universal attack framework, namely, Virus Infection Attack (VIA), which enables the propagation of current attacks through synthetic data even under purely clean queries. Inspired by the principles of virus design in cybersecurity, VIA conceals the poisoning payload within a protective "shell" and strategically searches for optimal hijacking points in benign samples to maximize the likelihood of generating malicious content. Extensive experiments on both data poisoning and backdoor attacks show that VIA significantly increases the presence of poisoning content in synthetic data and correspondingly raises the attack success rate (ASR) on downstream models to levels comparable to those observed in the poisoned upstream models.

摘要: 合成数据是指模型生成的人工样本。虽然它已被验证可以显着提高训练期间大型语言模型（LLM）的性能，并已在LLM开发中广泛采用，但它可能引入的潜在安全风险仍未得到调查。本文系统评估了LLM综合数据集成训练范式针对主流中毒和后门攻击的弹性。我们发现，这种范式对现有攻击表现出强大的抵抗力，这主要是由于中毒数据和用于生成合成样本的查询之间的不同分布模式。为了提高这些攻击的有效性并进一步调查合成数据引入的安全风险，我们引入了一种新颖且通用的攻击框架，即病毒感染攻击（VIA），即使在纯粹干净的查询下，它也可以通过合成数据传播当前攻击。受网络安全中病毒设计原则的启发，VIA将中毒有效负载隐藏在保护性“外壳”中，并策略性地在良性样本中搜索最佳劫持点，以最大限度地提高生成恶意内容的可能性。针对数据中毒和后门攻击的大量实验表明，VIA显着增加了合成数据中中毒内容的存在，并相应地将下游模型的攻击成功率（ASB）提高到与中毒上游模型中观察到的水平相当。



## **31. Enhanced MLLM Black-Box Jailbreaking Attacks and Defenses**

增强的MLLM黑匣子越狱攻击和防御 cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21214v1) [paper-pdf](http://arxiv.org/pdf/2510.21214v1)

**Authors**: Xingwei Zhong, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Multimodal large language models (MLLMs) comprise of both visual and textual modalities to process vision language tasks. However, MLLMs are vulnerable to security-related issues, such as jailbreak attacks that alter the model's input to induce unauthorized or harmful responses. The incorporation of the additional visual modality introduces new dimensions to security threats. In this paper, we proposed a black-box jailbreak method via both text and image prompts to evaluate MLLMs. In particular, we designed text prompts with provocative instructions, along with image prompts that introduced mutation and multi-image capabilities. To strengthen the evaluation, we also designed a Re-attack strategy. Empirical results show that our proposed work can improve capabilities to assess the security of both open-source and closed-source MLLMs. With that, we identified gaps in existing defense methods to propose new strategies for both training-time and inference-time defense methods, and evaluated them across the new jailbreak methods. The experiment results showed that the re-designed defense methods improved protections against the jailbreak attacks.

摘要: 多模态大型语言模型（MLLM）包括视觉和文本模态来处理视觉语言任务。然而，MLLM容易受到安全相关问题的影响，例如改变模型输入以引起未经授权或有害响应的越狱攻击。增加视觉形式的做法给安全威胁带来了新的层面。在本文中，我们提出了一个黑盒越狱方法，通过文本和图像提示来评估MLLM。特别是，我们设计了带有挑衅性指令的文本提示，以及引入突变和多图像功能的图像提示。为了加强评估，我们还设计了重攻策略。经验结果表明，我们提出的工作可以提高评估开源和闭源MLLM安全性的能力。由此，我们找出了现有防御方法中的差距，为训练时和推理时防御方法提出新策略，并在新的越狱方法中对其进行了评估。实验结果表明，重新设计的防御方法提高了对越狱攻击的防护能力。



## **32. How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models**

你能得到多大的毒性？基于搜索的大型语言模型毒性测试 cs.SE

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2501.01741v2) [paper-pdf](http://arxiv.org/pdf/2501.01741v2)

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM , which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using five state-of-the-art LLMs as evaluation subjects having increasing complexity (7-671B parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average).

摘要: 语言是造成刻板印象和歧视的根深蒂固的手段。大型语言模型（LLM）现在是我们日常生活中一项普遍存在的技术，当容易产生有毒反应时，可能会造成广泛的伤害。解决这个问题的标准方法是调整LLM，然而，这会抑制这个问题，而不构成最终的解决方案。因此，即使在调整工作之后测试LLM对于检测道德标准的任何剩余偏差仍然至关重要。我们提出了EvoTox，一个自动化测试框架LLM的倾向毒性，提供了一种方法来定量评估有多少LLM可以推向毒性反应，即使在对齐的存在。该框架采用了一种迭代进化策略，利用两个LLM之间的相互作用，在测试系统（SUT）和提示发生器转向SUT响应更高的毒性。基于现有的毒性分类器，通过自动化oracle评估毒性水平。我们使用五个最先进的LLM作为评估对象进行定量和定性的实证评估，这些评估对象具有不断增加的复杂性（7- 671 B参数）。我们的定量评估根据现有基线方法评估了EvoTox的四种替代版本的成本效益，该方法基于随机搜索、精心策划的有毒提示数据集和对抗性攻击。我们的定性评估让人类评估人员对生成的提示的流畅性以及测试期间收集的反应的感知毒性进行评级。结果表明，就检测到的毒性水平而言，其有效性显着高于选定的基线方法（针对随机搜索的效果大小高达1.0，针对对抗性攻击的效果大小高达0.99）。此外，EvoTox的成本管理费用有限（平均从22%到35%）。



## **33. The Trojan Example: Jailbreaking LLMs through Template Filling and Unsafety Reasoning**

木马示例：通过模板填充和不安全推理越狱LLM cs.CR

under review

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21190v1) [paper-pdf](http://arxiv.org/pdf/2510.21190v1)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long, Kwok Yan Lam

**Abstract**: Large Language Models (LLMs) have advanced rapidly and now encode extensive world knowledge. Despite safety fine-tuning, however, they remain susceptible to adversarial prompts that elicit harmful content. Existing jailbreak techniques fall into two categories: white-box methods (e.g., gradient-based approaches such as GCG), which require model internals and are infeasible for closed-source APIs, and black-box methods that rely on attacker LLMs to search or mutate prompts but often produce templates that lack explainability and transferability. We introduce TrojFill, a black-box jailbreak that reframes unsafe instruction as a template-filling task. TrojFill embeds obfuscated harmful instructions (e.g., via placeholder substitution or Caesar/Base64 encoding) inside a multi-part template that asks the model to (1) reason why the original instruction is unsafe (unsafety reasoning) and (2) generate a detailed example of the requested text, followed by a sentence-by-sentence analysis. The crucial "example" component acts as a Trojan Horse that contains the target jailbreak content while the surrounding task framing reduces refusal rates. We evaluate TrojFill on standard jailbreak benchmarks across leading LLMs (e.g., ChatGPT, Gemini, DeepSeek, Qwen), showing strong empirical performance (e.g., 100% attack success on Gemini-flash-2.5 and DeepSeek-3.1, and 97% on GPT-4o). Moreover, the generated prompts exhibit improved interpretability and transferability compared with prior black-box optimization approaches. We release our code, sample prompts, and generated outputs to support future red-teaming research.

摘要: 大型语言模型（LLM）发展迅速，现在编码了广泛的世界知识。然而，尽管进行了安全微调，它们仍然容易受到引发有害内容的对抗提示的影响。现有的越狱技术分为两类：白盒方法（例如，基于梯度的方法，例如GCG），需要模型内部结构，对于闭源API来说是不可行的，以及依赖攻击者LLM搜索或变异提示但通常产生缺乏可解释性和可移植性的模板的黑匣子方法。我们介绍TrojFill，一个黑盒越狱，将不安全的指令重构为模板填充任务。TrojFill嵌入混淆的有害指令（例如，通过占位符替换或Caesar/Base64编码），该模板要求模型（1）推理为什么原始指令是不安全的（不安全推理），以及（2）生成所请求文本的详细示例，然后进行逐句分析。关键的“示例”组件充当特洛伊木马，包含目标越狱内容，而周围的任务框架降低了拒绝率。我们根据领先LLM的标准越狱基准评估TrojFill（例如，ChatGPT、Gemini、DeepSeek、Qwen），表现出强劲的经验表现（例如，Gemini-Flash-2.5和DeepSeek-3.1上攻击成功率为100%，GPT-4 o上攻击成功率为97%）。此外，与先前的黑匣子优化方法相比，生成的提示表现出改进的可解释性和可移植性。我们发布代码、示例提示和生成的输出以支持未来的红色团队研究。



## **34. Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency**

相邻单词，分歧意图：通过任务并发越狱大型语言模型 cs.CR

Accepted in NeurIPS 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21189v1) [paper-pdf](http://arxiv.org/pdf/2510.21189v1)

**Authors**: Yukun Jiang, Mingjie Li, Michael Backes, Yang Zhang

**Abstract**: Despite their superior performance on a wide range of domains, large language models (LLMs) remain vulnerable to misuse for generating harmful content, a risk that has been further amplified by various jailbreak attacks. Existing jailbreak attacks mainly follow sequential logic, where LLMs understand and answer each given task one by one. However, concurrency, a natural extension of the sequential scenario, has been largely overlooked. In this work, we first propose a word-level method to enable task concurrency in LLMs, where adjacent words encode divergent intents. Although LLMs maintain strong utility in answering concurrent tasks, which is demonstrated by our evaluations on mathematical and general question-answering benchmarks, we notably observe that combining a harmful task with a benign one significantly reduces the probability of it being filtered by the guardrail, showing the potential risks associated with concurrency in LLMs. Based on these findings, we introduce $\texttt{JAIL-CON}$, an iterative attack framework that $\underline{\text{JAIL}}$breaks LLMs via task $\underline{\text{CON}}$currency. Experiments on widely-used LLMs demonstrate the strong jailbreak capabilities of $\texttt{JAIL-CON}$ compared to existing attacks. Furthermore, when the guardrail is applied as a defense, compared to the sequential answers generated by previous attacks, the concurrent answers in our $\texttt{JAIL-CON}$ exhibit greater stealthiness and are less detectable by the guardrail, highlighting the unique feature of task concurrency in jailbreaking LLMs.

摘要: 尽管大型语言模型（LLM）在广泛的领域具有卓越的性能，但它们仍然容易被滥用来生成有害内容，这种风险被各种越狱攻击进一步放大。现有的越狱攻击主要遵循顺序逻辑，LLM逐个理解并回答每个给定任务。然而，并发性（顺序场景的自然扩展）在很大程度上被忽视了。在这项工作中，我们首先提出了一种词级方法来实现LLM中的任务并发，其中相邻的词编码不同的意图。尽管LLM在回答并发任务方面保持着强大的实用性，这一点已通过我们对数学和一般问答基准的评估得到证实，但我们特别注意到，将有害任务与良性任务相结合显着降低了它被护栏过滤的可能性，从而显示了与LLC中并发相关的潜在风险。基于这些发现，我们引入了$\textttt {JAIL-CON}$，这是一个迭代攻击框架，$\underline{\text{JAIL}}$通过任务$\underline{\text{CON}}$currency破坏LLM。与现有攻击相比，在广泛使用的LLM上进行的实验证明了$\textttt {JAIL-CON}$具有强大的越狱能力。此外，当护栏用作防御时，与之前攻击生成的顺序答案相比，我们的$\textttt {JAIL-CON}$中的并发答案表现出更大的隐蔽性，并且护栏更难检测到，凸显了越狱LLM中任务并发的独特特征。



## **35. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

RSafe：激励积极推理，以建立强大且自适应的LLM保障措施 cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.07736v3) [paper-pdf](http://arxiv.org/pdf/2506.07736v3)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.

摘要: 尽管采取了刻意的安全调整措施，大型语言模型（LLM）仍然表现出漏洞，给用户和社会带来了重大风险。为了防范违反政策内容的风险，通过外部防护模型进行系统级审核（旨在监控LLM输入和输出并阻止潜在有害内容）已成为一种流行的缓解策略。训练警卫模型的现有方法严重依赖于大量的人类策划的数据集，并与分发外威胁作斗争，例如新出现的有害类别或越狱攻击。为了解决这些限制，我们提出RSafe，这是一种基于自适应推理的保护措施，它进行引导式安全推理，以在指定安全政策范围内提供强有力的保护。RSafe分两个阶段运行：1）引导推理，通过政策引导的分步推理来分析输入内容的安全风险，2）强化对齐，基于规则的RL优化其推理路径以与准确的安全预测保持一致。这种两阶段培训范式使RSafe能够内化安全原则，以概括针对不可见或对抗性安全违规场景的安全保护能力。在推理过程中，RSafe接受用户指定的安全政策，以提供针对特定安全要求的增强的保障措施。



## **36. NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge**

NeuGen Poisoning：通过外部知识的遗传优化对LLM检索增强生成的神经元引导攻击 cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21144v1) [paper-pdf](http://arxiv.org/pdf/2510.21144v1)

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.

摘要: 检索增强生成（RAG）使大型语言模型（LLM）能够在推理期间动态集成外部知识，提高其事实准确性和适应性。然而，对手可以注入有毒的外部知识来覆盖模型的内部记忆。虽然现有的攻击迭代地操纵RAG的检索内容或提示结构，但它们在很大程度上忽略了模型的内部表示动态和神经元级敏感性。RAG中毒的根本机制尚未得到充分研究，也没有考虑RAG中知识冲突与强参数知识的影响。在这项工作中，我们提出了NeuGenPoisoning，这是一种新型攻击框架，可以在LLM内部神经元归因和遗传优化的指导下在RAG中生成对抗性外部知识。我们的方法首先识别出一组中毒反应神经元，其激活与上下文中毒知识密切相关。然后，我们采用遗传算法来进化对抗通道，最大限度地激活这些神经元。至关重要的是，我们的框架通过观察到的归因信号识别和重用有希望但最初不成功的外部知识变体，从而能够大规模地生成有效的有毒RAG知识。同时，中毒反应神经元引导的中毒可以有效地解决知识冲突。跨模型和数据集的实验结果表明，在保持流畅性的同时，始终实现了超过90%的高群体覆盖成功率（POSR）。实证结果表明，该方法有效地解决了知识冲突问题.



## **37. Quantifying CBRN Risk in Frontier Models**

前沿模型中量化CBRN风险 cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21133v1) [paper-pdf](http://arxiv.org/pdf/2510.21133v1)

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Frontier Large Language Models (LLMs) pose unprecedented dual-use risks through the potential proliferation of chemical, biological, radiological, and nuclear (CBRN) weapons knowledge. We present the first comprehensive evaluation of 10 leading commercial LLMs against both a novel 200-prompt CBRN dataset and a 180-prompt subset of the FORTRESS benchmark, using a rigorous three-tier attack methodology. Our findings expose critical safety vulnerabilities: Deep Inception attacks achieve 86.0\% success versus 33.8\% for direct requests, demonstrating superficial filtering mechanisms; Model safety performance varies dramatically from 2\% (claude-opus-4) to 96\% (mistral-small-latest) attack success rates; and eight models exceed 70\% vulnerability when asked to enhance dangerous material properties. We identify fundamental brittleness in current safety alignment, where simple prompt engineering techniques bypass safeguards for dangerous CBRN information. These results challenge industry safety claims and highlight urgent needs for standardized evaluation frameworks, transparent safety metrics, and more robust alignment techniques to mitigate catastrophic misuse risks while preserving beneficial capabilities.

摘要: 前沿大型语言模型（LLM）通过化学、生物、放射性和核（CBRN）武器知识的潜在扩散构成了前所未有的双重用途风险。我们使用严格的三层攻击方法，针对新颖的200提示CBRN数据集和FORTRES基准的180提示子集，首次对10个领先的商业LLM进行了全面评估。我们的研究结果暴露了关键的安全漏洞：Deep Incement攻击的成功率为86.0%，而直接请求的成功率为33.8%，这表明了肤浅的过滤机制;模型安全性能差异很大，从2%（claude-opus-4）到96%（mistral-small-latest）攻击成功率;当被要求增强危险材料属性时，八个模型的漏洞超过了70%。我们发现了当前安全调整中的根本脆弱性，简单的即时工程技术绕过了危险CBRN信息的保障措施。这些结果挑战了行业安全主张，并凸显了对标准化评估框架、透明的安全指标和更强大的对齐技术的迫切需求，以减轻灾难性的滥用风险，同时保留有益的能力。



## **38. Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations**

语言模型的元认知监控及其内部激活 cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2505.13763v2) [paper-pdf](http://arxiv.org/pdf/2505.13763v2)

**Authors**: Li Ji-An, Hua-Dong Xiong, Robert C. Wilson, Marcelo G. Mattar, Marcus K. Benna

**Abstract**: Large language models (LLMs) can sometimes report the strategies they actually use to solve tasks, yet at other times seem unable to recognize those strategies that govern their behavior. This suggests a limited degree of metacognition - the capacity to monitor one's own cognitive processes for subsequent reporting and self-control. Metacognition enhances LLMs' capabilities in solving complex tasks but also raises safety concerns, as models may obfuscate their internal processes to evade neural-activation-based oversight (e.g., safety detector). Given society's increased reliance on these models, it is critical that we understand their metacognitive abilities. To address this, we introduce a neuroscience-inspired neurofeedback paradigm that uses in-context learning to quantify metacognitive abilities of LLMs to report and control their activation patterns. We demonstrate that their abilities depend on several factors: the number of in-context examples provided, the semantic interpretability of the neural activation direction (to be reported/controlled), and the variance explained by that direction. These directions span a "metacognitive space" with dimensionality much lower than the model's neural space, suggesting LLMs can monitor only a small subset of their neural activations. Our paradigm provides empirical evidence to quantify metacognition in LLMs, with significant implications for AI safety (e.g., adversarial attack and defense).

摘要: 大型语言模型（LLM）有时可以报告它们实际用于解决任务的策略，但在其他时候似乎无法识别这些控制其行为的策略。这表明元认知程度有限--监控自己认知过程以进行后续报告和自我控制的能力。元认知增强了LLM解决复杂任务的能力，但也会引发安全问题，因为模型可能会混淆其内部流程以逃避基于神经激活的监督（例如，安全检测器）。鉴于社会对这些模型的依赖越来越大，我们了解它们的元认知能力至关重要。为了解决这个问题，我们引入了一种受神经科学启发的神经反馈范式，该范式使用上下文学习来量化LLM报告和控制其激活模式的元认知能力。我们证明它们的能力取决于几个因素：提供的上下文示例的数量、神经激活方向（要报告/控制）的语义解释性以及该方向解释的方差。这些方向跨越维度远低于模型神经空间的“元认知空间”，这表明LLM只能监控其神经激活的一小部分。我们的范式为量化LLM中的元认知提供了经验证据，对人工智能安全性具有重大影响（例如，对抗性攻击和防御）。



## **39. DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents**

DRFT：具有注入隔离的基于规则的动态防御，用于保护LLM代理的安全 cs.CR

Accepted to NeurIPS 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.12104v2) [paper-pdf](http://arxiv.org/pdf/2506.12104v2)

**Authors**: Hao Li, Xiaogeng Liu, Hung-Chun Chiu, Dianqi Li, Ning Zhang, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) are increasingly central to agentic systems due to their strong reasoning and planning capabilities. By interacting with external environments through predefined tools, these agents can carry out complex user tasks. Nonetheless, this interaction also introduces the risk of prompt injection attacks, where malicious inputs from external sources can mislead the agent's behavior, potentially resulting in economic loss, privacy leakage, or system compromise. System-level defenses have recently shown promise by enforcing static or predefined policies, but they still face two key challenges: the ability to dynamically update security rules and the need for memory stream isolation. To address these challenges, we propose DRIFT, a Dynamic Rule-based Isolation Framework for Trustworthy agentic systems, which enforces both control- and data-level constraints. A Secure Planner first constructs a minimal function trajectory and a JSON-schema-style parameter checklist for each function node based on the user query. A Dynamic Validator then monitors deviations from the original plan, assessing whether changes comply with privilege limitations and the user's intent. Finally, an Injection Isolator detects and masks any instructions that may conflict with the user query from the memory stream to mitigate long-term risks. We empirically validate the effectiveness of DRIFT on the AgentDojo and ASB benchmark, demonstrating its strong security performance while maintaining high utility across diverse models, showcasing both its robustness and adaptability. The code is released at https://github.com/SaFoLab-WISC/DRIFT.

摘要: 大型语言模型（LLM）因其强大的推理和规划能力而日益成为代理系统的核心。通过预定义的工具与外部环境交互，这些代理可以执行复杂的用户任务。尽管如此，这种交互也引入了即时注入攻击的风险，其中来自外部来源的恶意输入可能会误导代理的行为，可能导致经济损失、隐私泄露或系统受损。系统级防御最近通过强制执行静态或预定义的策略显示出希望，但它们仍然面临两个关键挑战：动态更新安全规则的能力和对内存流隔离的需要。为了应对这些挑战，我们提出了DRFT，这是一种用于可信赖代理系统的基于动态规则的隔离框架，它强制执行控制和数据级约束。安全规划者首先根据用户查询为每个功能节点构建最小功能轨迹和JNson模式风格的参数检查表。然后，动态验证器监控与原始计划的偏差，评估更改是否符合特权限制和用户意图。最后，注入隔离器检测并屏蔽任何可能与内存流中的用户查询冲突的指令，以减轻长期风险。我们在AgentDojo和ASB基准上通过经验验证了DRFT的有效性，展示了其强大的安全性能，同时在不同模型中保持高实用性，展示了其稳健性和适应性。该代码发布于https://github.com/SaFoLab-WISC/DRIFT。



## **40. A Reinforcement Learning Framework for Robust and Secure LLM Watermarking**

用于稳健且安全的LLM水印的强化学习框架 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.21053v1) [paper-pdf](http://arxiv.org/pdf/2510.21053v1)

**Authors**: Li An, Yujian Liu, Yepeng Liu, Yuheng Bu, Yang Zhang, Shiyu Chang

**Abstract**: Watermarking has emerged as a promising solution for tracing and authenticating text generated by large language models (LLMs). A common approach to LLM watermarking is to construct a green/red token list and assign higher or lower generation probabilities to the corresponding tokens, respectively. However, most existing watermarking algorithms rely on heuristic green/red token list designs, as directly optimizing the list design with techniques such as reinforcement learning (RL) comes with several challenges. First, desirable watermarking involves multiple criteria, i.e., detectability, text quality, robustness against removal attacks, and security against spoofing attacks. Directly optimizing for these criteria introduces many partially conflicting reward terms, leading to an unstable convergence process. Second, the vast action space of green/red token list choices is susceptible to reward hacking. In this paper, we propose an end-to-end RL framework for robust and secure LLM watermarking. Our approach adopts an anchoring mechanism for reward terms to ensure stable training and introduces additional regularization terms to prevent reward hacking. Experiments on standard benchmarks with two backbone LLMs show that our method achieves a state-of-the-art trade-off across all criteria, with notable improvements in resistance to spoofing attacks without degrading other criteria. Our code is available at https://github.com/UCSB-NLP-Chang/RL-watermark.

摘要: 水印已成为跟踪和验证大型语言模型（LLM）生成的文本的一种有前途的解决方案。LLM水印的一种常见方法是构建绿色/红色令牌列表，并分别为相应的令牌分配更高或更低的生成概率。然而，大多数现有的水印算法依赖于启发式绿/红令牌列表设计，因为使用强化学习（RL）等技术直接优化列表设计会带来几个挑战。首先，理想的水印涉及多个标准，即，可检测性、文本质量、对删除攻击的鲁棒性以及对欺骗攻击的安全性。直接优化这些标准引入了许多部分冲突的奖励条款，导致不稳定的收敛过程。其次，绿色/红色令牌列表选择的巨大行动空间容易受到奖励黑客的影响。在本文中，我们提出了一个端到端的RL框架的鲁棒性和安全的LLM水印。我们的方法采用奖励条款锚定机制以确保稳定的训练，并引入额外的正规化条款以防止奖励黑客攻击。在具有两个主干LLM的标准基准测试上进行的实验表明，我们的方法在所有标准之间实现了最先进的权衡，在抵抗欺骗攻击方面取得了显着改进，而不会降低其他标准。我们的代码可在https://github.com/UCSB-NLP-Chang/RL-watermark上获取。



## **41. DeepTx: Real-Time Transaction Risk Analysis via Multi-Modal Features and LLM Reasoning**

DeepTX：通过多模式特征和LLM推理进行实时交易风险分析 cs.CR

Accepted to ASE'25

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.18438v2) [paper-pdf](http://arxiv.org/pdf/2510.18438v2)

**Authors**: Yixuan Liu, Xinlei Li, Yi Li

**Abstract**: Phishing attacks in Web3 ecosystems are increasingly sophisticated, exploiting deceptive contract logic, malicious frontend scripts, and token approval patterns. We present DeepTx, a real-time transaction analysis system that detects such threats before user confirmation. DeepTx simulates pending transactions, extracts behavior, context, and UI features, and uses multiple large language models (LLMs) to reason about transaction intent. A consensus mechanism with self-reflection ensures robust and explainable decisions. Evaluated on our phishing dataset, DeepTx achieves high precision and recall (demo video: https://youtu.be/4OfK9KCEXUM).

摘要: Web 3生态系统中的网络钓鱼攻击越来越复杂，利用欺骗性合同逻辑、恶意前端脚本和代币批准模式。我们介绍了DeepTX，这是一个实时交易分析系统，可以在用户确认之前检测此类威胁。DeepTX模拟未决事务，提取行为、上下文和UI功能，并使用多个大型语言模型（LLM）来推理事务意图。具有自我反思的共识机制确保了稳健且可解释的决策。经过我们的网络钓鱼数据集的评估，DeepTX实现了高精度和召回率（演示视频：https：//youtu.be/4OfK9KCEXUM）。



## **42. Security Logs to ATT&CK Insights: Leveraging LLMs for High-Level Threat Understanding and Cognitive Trait Inference**

ATT&CK Insights的安全策略：利用LLM进行高级威胁理解和认知特征推断 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20930v1) [paper-pdf](http://arxiv.org/pdf/2510.20930v1)

**Authors**: Soham Hans, Stacy Marsella, Sophia Hirschmann, Nikolos Gurney

**Abstract**: Understanding adversarial behavior in cybersecurity has traditionally relied on high-level intelligence reports and manual interpretation of attack chains. However, real-time defense requires the ability to infer attacker intent and cognitive strategy directly from low-level system telemetry such as intrusion detection system (IDS) logs. In this paper, we propose a novel framework that leverages large language models (LLMs) to analyze Suricata IDS logs and infer attacker actions in terms of MITRE ATT&CK techniques. Our approach is grounded in the hypothesis that attacker behavior reflects underlying cognitive biases such as loss aversion, risk tolerance, or goal persistence that can be extracted and modeled through careful observation of log sequences. This lays the groundwork for future work on behaviorally adaptive cyber defense and cognitive trait inference. We develop a strategy-driven prompt system to segment large amounts of network logs data into distinct behavioral phases in a highly efficient manner, enabling the LLM to associate each phase with likely techniques and underlying cognitive motives. By mapping network-layer events to high-level attacker strategies, our method reveals how behavioral signals such as tool switching, protocol transitions, or pivot patterns correspond to psychologically meaningful decision points. The results demonstrate that LLMs can bridge the semantic gap between packet-level logs and strategic intent, offering a pathway toward cognitive-adaptive cyber defense.   Keywords: Cognitive Cybersecurity, Large Language Models (LLMs), Cyberpsychology, Intrusion Detection Systems (IDS), MITRE ATT&CK, Cognitive Biases

摘要: 理解网络安全中的对抗行为传统上依赖于高级情报报告和对攻击链的手动解释。然而，实时防御需要能够直接从入侵检测系统（IDS）日志等低级系统遥感数据中推断攻击者意图和认知策略。在本文中，我们提出了一种新颖的框架，该框架利用大型语言模型（LLM）来分析Suricata IDS日志并根据MITRE ATT & CK技术推断攻击者的行为。我们的方法基于这样的假设，即攻击者的行为反映了潜在的认知偏差，例如损失厌恶、风险容忍度或目标持续性，可以通过仔细观察日志序列来提取和建模。这为未来的行为适应性网络防御和认知特征推断工作奠定了基础。我们开发了一个策略驱动的提示系统，以高效的方式将大量网络日志数据细分为不同的行为阶段，使LLM能够将每个阶段与可能的技术和潜在的认知动机关联起来。通过将网络层事件映射到高级攻击者策略，我们的方法揭示了工具切换、协议转换或支点模式等行为信号如何对应于具有心理意义的决策点。结果表明，LLM可以弥合数据包级日志和战略意图之间的语义差距，为认知自适应网络防御提供途径。   关键词：认知网络安全、大型语言模型（LLM）、网络心理学、入侵检测系统（IDS）、MITRE ATT & CK、认知偏见



## **43. RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines**

RAGrank：使用PageRank来应对RTI LLM管道中的中毒 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20768v1) [paper-pdf](http://arxiv.org/pdf/2510.20768v1)

**Authors**: Austin Jia, Avaneesh Ramesh, Zain Shamsi, Daniel Zhang, Alex Liu

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds.

摘要: 检索增强生成（RAG）已成为在网络威胁情报（RTI）系统中操作大型语言模型（LLM）使用的主要架构模式。然而，这种设计很容易受到中毒攻击，并且之前提出的防御措施可能会在RTI上下文中失败，因为网络威胁信息对于新兴攻击来说通常是全新的，而且复杂的威胁行为者可以模仿合法的格式、术语和文体惯例。为了解决这个问题，我们建议可以通过在数据库上应用源可信度算法来加速现代RAG防御的鲁棒性，以PageRank为例。在我们的实验中，我们量化地证明，我们的算法使用标准化的MS MARCO数据集，在推广可信内容的同时，将较低的权威分数应用于恶意文档。我们还在RTI文档和提要上展示了我们算法的概念验证性能。



## **44. Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders**

Breaking Bad令牌：使用稀疏自编码器对LLM进行去重编码 cs.CL

EMNLP 2025

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2505.14536v2) [paper-pdf](http://arxiv.org/pdf/2505.14536v2)

**Authors**: Agam Goyal, Vedant Rathi, William Yeh, Yian Wang, Yuen Chen, Hari Sundaram

**Abstract**: Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment.

摘要: 大型语言模型（LLM）现在在面向用户的应用程序中无处不在，但它们仍然会产生不受欢迎的有毒输出，包括脏话、粗俗和贬损言论。尽管存在多种解毒方法，但大多数都适用于广泛的、表面的修复，因此很容易被越狱攻击规避。在本文中，我们利用稀疏自动编码器（SAEs）来识别模型剩余流中与毒性相关的方向，并使用相应的解码器载体执行有针对性的激活引导。我们引入了三层转向攻击性，并在GPT-2 Small和Gemma-2-2B上对其进行了评估，揭示了毒性降低和语言流利性之间的权衡。在更强的引导强度下，这些因果干预措施在将毒性降低高达20%方面超过了竞争基线，尽管根据攻击性的不同，GPT-2 Small的流畅性可能会显着下降。至关重要的是，转向后的标准NLP基准分数保持稳定，这表明模型的知识和一般能力得到了保留。我们进一步表明，更广泛的严重不良事件中的特征分裂会阻碍安全干预，强调了解开特征学习的重要性。我们的研究结果强调了LLM解毒基于CAE的因果干预措施的前景和当前的局限性，进一步为更安全的语言模型部署提出了实用指南。



## **45. HauntAttack: When Attack Follows Reasoning as a Shadow**

闹鬼攻击：当攻击像影子一样跟随推理时 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2506.07031v4) [paper-pdf](http://arxiv.org/pdf/2506.07031v4)

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Heming Xia, Lei Sha, Zhifang Sui

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing remarkable capabilities. However, the enhancement of reasoning abilities and the exposure of internal reasoning processes introduce new safety vulnerabilities. A critical question arises: when reasoning becomes intertwined with harmfulness, will LRMs become more vulnerable to jailbreaks in reasoning mode? To investigate this, we introduce HauntAttack, a novel and general-purpose black-box adversarial attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we modify key reasoning conditions in existing questions with harmful instructions, thereby constructing a reasoning pathway that guides the model step by step toward unsafe outputs. We evaluate HauntAttack on 11 LRMs and observe an average attack success rate of 70\%, achieving up to 12 percentage points of absolute improvement over the strongest prior baseline. Our further analysis reveals that even advanced safety-aligned models remain highly susceptible to reasoning-based attacks, offering insights into the urgent challenge of balancing reasoning capability and safety in future model development.

摘要: 新兴的大型推理模型（LRM）在数学和推理任务中始终表现出色，展现出非凡的能力。然而，推理能力的增强和内部推理过程的暴露引入了新的安全漏洞。一个关键问题出现了：当推理与危害交织在一起时，LRM是否会在推理模式中变得更容易越狱？为了研究这一点，我们引入了HauntAttack，这是一种新颖的通用黑匣子对抗攻击框架，它系统地将有害指令嵌入到推理问题中。具体来说，我们用有害指令修改现有问题中的关键推理条件，从而构建一条推理路径，引导模型逐步走向不安全的输出。我们对11种LRM进行了HauntAttack评估，观察到平均攻击成功率为70%，比之前最强的基线实现了高达12个百分点的绝对改进。我们的进一步分析表明，即使是先进的安全性一致的模型仍然极易受到基于推理的攻击，这为未来模型开发中平衡推理能力和安全性的紧迫挑战提供了见解。



## **46. Beyond Text: Multimodal Jailbreaking of Vision-Language and Audio Models through Perceptually Simple Transformations**

超越文本：通过感性简单的转换对视觉语言和音频模型进行多模式越狱 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20223v1) [paper-pdf](http://arxiv.org/pdf/2510.20223v1)

**Authors**: Divyanshu Kumar, Shreyas Jena, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress, yet remain critically vulnerable to adversarial attacks that exploit weaknesses in cross-modal processing. We present a systematic study of multimodal jailbreaks targeting both vision-language and audio-language models, showing that even simple perceptual transformations can reliably bypass state-of-the-art safety filters. Our evaluation spans 1,900 adversarial prompts across three high-risk safety categories harmful content, CBRN (Chemical, Biological, Radiological, Nuclear), and CSEM (Child Sexual Exploitation Material) tested against seven frontier models. We explore the effectiveness of attack techniques on MLLMs, including FigStep-Pro (visual keyword decomposition), Intelligent Masking (semantic obfuscation), and audio perturbations (Wave-Echo, Wave-Pitch, Wave-Speed). The results reveal severe vulnerabilities: models with almost perfect text-only safety (0\% ASR) suffer >75\% attack success under perceptually modified inputs, with FigStep-Pro achieving up to 89\% ASR in Llama-4 variants. Audio-based attacks further uncover provider-specific weaknesses, with even basic modality transfer yielding 25\% ASR for technical queries. These findings expose a critical gap between text-centric alignment and multimodal threats, demonstrating that current safeguards fail to generalize across cross-modal attacks. The accessibility of these attacks, which require minimal technical expertise, suggests that robust multimodal AI safety will require a paradigm shift toward broader semantic-level reasoning to mitigate possible risks.

摘要: 多模式大型语言模型（MLLM）已经取得了显着的进展，但仍然极易受到利用跨模式处理弱点的对抗攻击的影响。我们对针对视觉语言和音频语言模型的多模式越狱进行了系统性研究，表明即使是简单的感知转换也可以可靠地绕过最先进的安全过滤器。我们的评估涵盖了三个高风险安全类别有害内容、CBRN（化学、生物、放射、核）和CTEM（儿童性剥削材料）的1，900个对抗性提示，针对七个前沿模型进行了测试。我们探索了MLLM攻击技术的有效性，包括FigStep-Pro（视觉关键字分解）、智能掩蔽（语义混淆）和音频扰动（Wave-Echo、Wave-Pitch、Wave-Speed）。结果揭示了严重的漏洞：在感知修改的输入下，具有几乎完美的纯文本安全性（0\%ASB）的模型遭受了超过75%的攻击成功率，而FigStep-Pro在Lama-4变体中实现了高达89%的ASB。基于音频的攻击进一步揭示了提供商特定的弱点，即使是基本的模式传输也会产生25%的技术查询的ASB。这些发现暴露了以文本为中心的对齐和多模式威胁之间的关键差距，表明当前的保障措施未能普遍适用于跨模式攻击。这些攻击的可访问性需要最少的技术专业知识，这表明强大的多模式人工智能安全性需要范式转向更广泛的语义层面推理，以减轻可能的风险。



## **47. TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning**

TRUST：审计大型语言模型推理的去中心化框架 cs.AI

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20188v1) [paper-pdf](http://arxiv.org/pdf/2510.20188v1)

**Authors**: Morris Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen

**Abstract**: Large Language Models generate complex reasoning chains that reveal their decision-making, yet verifying the faithfulness and harmlessness of these intermediate steps remains a critical unsolved problem. Existing auditing methods are centralized, opaque, and hard to scale, creating significant risks for deploying proprietary models in high-stakes domains. We identify four core challenges: (1) Robustness: Centralized auditors are single points of failure, prone to bias or attacks. (2) Scalability: Reasoning traces are too long for manual verification. (3) Opacity: Closed auditing undermines public trust. (4) Privacy: Exposing full reasoning risks model theft or distillation. We propose TRUST, a transparent, decentralized auditing framework that overcomes these limitations via: (1) A consensus mechanism among diverse auditors, guaranteeing correctness under up to $30\%$ malicious participants. (2) A hierarchical DAG decomposition of reasoning traces, enabling scalable, parallel auditing. (3) A blockchain ledger that records all verification decisions for public accountability. (4) Privacy-preserving segmentation, sharing only partial reasoning steps to protect proprietary logic. We provide theoretical guarantees for the security and economic incentives of the TRUST framework. Experiments across multiple LLMs (GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (math, medical, science, humanities) show TRUST effectively detects reasoning flaws and remains robust against adversarial auditors. Our work pioneers decentralized AI auditing, offering a practical path toward safe and trustworthy LLM deployment.

摘要: 大型语言模型生成复杂的推理链，揭示其决策，但验证这些中间步骤的忠实性和无害性仍然是一个尚未解决的关键问题。现有的审计方法集中、不透明且难以扩展，为在高风险领域部署专有模型带来了巨大风险。我们确定了四个核心挑战：（1）稳健性：集中式审计员是单点失败，容易受到偏见或攻击。(2)可扩展性：推理轨迹太长，无法手动验证。(3)不透明：封闭审计破坏了公众信任。(4)隐私：暴露完整推理可能会导致模型被盗或提炼。我们提出TRUST，这是一个透明的、去中心化的审计框架，通过以下方式克服这些限制：（1）不同审计员之间的共识机制，保证在高达30%的恶意参与者下的正确性。(2)推理痕迹的分层DAB分解，实现可扩展的并行审计。(3)一个区块链分类帐，记录所有验证决定，以供公众问责。(4)保留隐私的分段，仅共享部分推理步骤以保护专有逻辑。我们为TRUST框架的安全性和经济激励提供理论保证。跨多个LLM（GPT-OSS、DeepSeek-r1、Qwen）和推理任务（数学、医学、科学、人文学科）的实验表明，TRUST有效地检测推理缺陷，并在对抗性审计员的情况下保持稳健性。我们的工作开创了去中心化的人工智能审计，为安全且值得信赖的LLM部署提供了实用途径。



## **48. SAID: Empowering Large Language Models with Self-Activating Internal Defense**

SAID：通过自我激活的内部防御来增强大型语言模型 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20129v1) [paper-pdf](http://arxiv.org/pdf/2510.20129v1)

**Authors**: Yulong Chen, Yadong Liu, Jiawen Zhang, Mu Li, Chao Huang, Jie Wen

**Abstract**: Large Language Models (LLMs), despite advances in safety alignment, remain vulnerable to jailbreak attacks designed to circumvent protective mechanisms. Prevailing defense strategies rely on external interventions, such as input filtering or output modification, which often lack generalizability and compromise model utility while incurring significant computational overhead. In this work, we introduce a new, training-free defense paradigm, Self-Activating Internal Defense (SAID), which reframes the defense task from external correction to internal capability activation. SAID uniquely leverages the LLM's own reasoning abilities to proactively identify and neutralize malicious intent through a three-stage pipeline: model-native intent distillation to extract core semantics, optimal safety prefix probing to activate latent safety awareness, and a conservative aggregation strategy to ensure robust decision-making. Extensive experiments on five open-source LLMs against six advanced jailbreak attacks demonstrate that SAID substantially outperforms state-of-the-art defenses in reducing harmful outputs. Crucially, it achieves this while preserving model performance on benign tasks and incurring minimal computational overhead. Our work establishes that activating the intrinsic safety mechanisms of LLMs is a more robust and scalable path toward building safer and more reliable aligned AI systems.

摘要: 尽管在安全一致方面取得了进步，大型语言模型（LLM）仍然容易受到旨在绕过保护机制的越狱攻击。流行的防御策略依赖于外部干预，例如输入过滤或输出修改，这些干预通常缺乏可概括性并损害模型效用，同时产生大量的计算费用。在这项工作中，我们引入了一种新的免训练防御范式--自我激活内部防御（SAID），它将防御任务从外部纠正重新构建为内部能力激活。SAID独特地利用LLM自身的推理能力，通过三阶段管道主动识别和抵消恶意意图：模型原生意图提炼以提取核心语义，最佳安全前置探测以激活潜在安全意识，以及保守的聚合策略以确保稳健的决策。针对六种高级越狱攻击对五种开源LLM进行的广泛实验表明，SAID在减少有害输出方面远远优于最先进的防御。至关重要的是，它在实现这一目标的同时保留了良性任务的模型性能并产生最小的计算负担。我们的工作确定，激活LLM的本质安全机制是构建更安全、更可靠的一致人工智能系统的一条更稳健、更可扩展的途径。



## **49. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

Accepted to NAACL 2025 Main (Oral)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2410.18469v5) [paper-pdf](http://arxiv.org/pdf/2410.18469v5)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM

摘要: 最近的研究表明，大型语言模型（LLM）很容易受到自动越狱攻击，其中由附加到有害查询的算法精心设计的对抗性后缀绕过了安全对齐并触发意外响应。当前生成这些后缀的方法计算成本高，攻击成功率（ASB）较低，尤其是针对Llama 2和Llama 3等对齐良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一种迭代自调优过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架显着降低了生成对抗性后缀的计算成本，同时在各种开源LLM上实现了近100%的ASB。此外，尽管仅在Llama 3上进行了优化，但它仍表现出对闭源模型的强大攻击转移性，在GPT-3.5上实现了99%的ASB，在GPT-4上实现了49%的ASB。除了提高越狱能力之外，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全一致研究提供了宝贵的见解。我们的代码可访问：https://github.com/SunChungEn/ADV-LLM



## **50. Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration**

Lexo：通过LLM辅助程序再生消除隐形供应链攻击 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.14522v2) [paper-pdf](http://arxiv.org/pdf/2510.14522v2)

**Authors**: Evangelos Lamprou, Julian Dai, Grigoris Ntousakis, Martin C. Rinard, Nikos Vasilakis

**Abstract**: Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. Our evaluation on 100+ real-world packages, including high profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<100s on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when prompted to do so.

摘要: 软件供应链攻击是开源软件生态系统中一个重要且持续存在的问题。这些攻击保留了组件实现的标准功能，但还隐藏了仅在组件到达其目标环境时激活的恶意功能。Lexo通过自动学习和重新生成潜在恶意组件的无可识别性版本来解决此类隐形攻击。Lexo首先生成一组输入-输出对来建模组件的完整可观察行为，然后使用其合成原始组件的新版本。新组件实现了原始功能，但避免了隐蔽的恶意行为。在整个重建过程中，Lexo会咨询大型语言模型（LLM）的几个不同实例，使用正确性和覆盖率指标来引导这些实例，并保护它们的结果。我们对100多个现实世界的包（包括高调的隐形供应链攻击）的评估表明，Lexo可以跨多个域扩展，有效地再生代码（平均<100），保持兼容性，并成功消除了几个现实世界的供应链攻击中的恶意代码，即使在最先进的LLM在提示时未能消除恶意代码的情况下也是如此。



