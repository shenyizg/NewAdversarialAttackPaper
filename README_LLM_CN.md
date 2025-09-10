# Latest Large Language Model Attack Papers
**update at 2025-09-10 10:56:29**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. ImportSnare: Directed "Code Manual" Hijacking in Retrieval-Augmented Code Generation**

ImportSnare：检索增强代码生成中的定向“代码手册”劫持 cs.CR

This paper has been accepted by the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07941v1) [paper-pdf](http://arxiv.org/pdf/2509.07941v1)

**Authors**: Kai Ye, Liangcai Su, Chenxiong Qian

**Abstract**: Code generation has emerged as a pivotal capability of Large Language Models(LLMs), revolutionizing development efficiency for programmers of all skill levels. However, the complexity of data structures and algorithmic logic often results in functional deficiencies and security vulnerabilities in generated code, reducing it to a prototype requiring extensive manual debugging. While Retrieval-Augmented Generation (RAG) can enhance correctness and security by leveraging external code manuals, it simultaneously introduces new attack surfaces.   In this paper, we pioneer the exploration of attack surfaces in Retrieval-Augmented Code Generation (RACG), focusing on malicious dependency hijacking. We demonstrate how poisoned documentation containing hidden malicious dependencies (e.g., matplotlib_safe) can subvert RACG, exploiting dual trust chains: LLM reliance on RAG and developers' blind trust in LLM suggestions. To construct poisoned documents, we propose ImportSnare, a novel attack framework employing two synergistic strategies: 1)Position-aware beam search optimizes hidden ranking sequences to elevate poisoned documents in retrieval results, and 2)Multilingual inductive suggestions generate jailbreaking sequences to manipulate LLMs into recommending malicious dependencies. Through extensive experiments across Python, Rust, and JavaScript, ImportSnare achieves significant attack success rates (over 50% for popular libraries such as matplotlib and seaborn) in general, and is also able to succeed even when the poisoning ratio is as low as 0.01%, targeting both custom and real-world malicious packages. Our findings reveal critical supply chain risks in LLM-powered development, highlighting inadequate security alignment for code generation tasks. To support future research, we will release the multilingual benchmark suite and datasets. The project homepage is https://importsnare.github.io.

摘要: 代码生成已成为大型语言模型（LLM）的关键功能，彻底改变了所有技能水平的程序员的开发效率。然而，数据结构和算法逻辑的复杂性往往会导致生成的代码中存在功能缺陷和安全漏洞，从而将其简化为需要大量手动调试的原型。虽然检索增强生成（RAG）可以通过利用外部代码手册来增强正确性和安全性，但它同时引入了新的攻击表面。   在本文中，我们率先探索检索增强代码生成（RACG）中的攻击表面，重点关注恶意依赖劫持。我们演示了如何包含隐藏恶意依赖项的有毒文档（例如，matplotlib_safe）可以利用双重信任链颠覆RACG：LLM对RAG的依赖以及开发人员对LLM建议的盲目信任。为了构建有毒文档，我们提出了ImportSnare，这是一种采用两种协同策略的新型攻击框架：1）位置感知射束搜索优化隐藏排名序列，以提升检索结果中的有毒文档，2）多语言归纳建议生成越狱序列，以操纵LLM推荐恶意依赖项。通过Python、Rust和JavaScript的广泛实验，ImportSnare总体上实现了显着的攻击成功率（matplotlib和seaborn等流行库超过50%），并且即使中毒率低至0.01%，也能够成功，针对自定义和现实世界的恶意软件包。我们的研究结果揭示了LLM驱动的开发中的关键供应链风险，凸显了代码生成任务的安全一致性不足。为了支持未来的研究，我们将发布多语言基准套件和数据集。项目主页为https://importsnare.github.io。



## **2. Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees**

使用结构化攻击树的LLM驱动渗透测试中的引导推理 cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07939v1) [paper-pdf](http://arxiv.org/pdf/2509.07939v1)

**Authors**: Katsuaki Nakano, Reza Feyyazi, Shanchieh Jay Yang, Michael Zuzak

**Abstract**: Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments

摘要: 大型语言模型（LLM）的最新进展激发了人们对自动化网络安全渗透测试工作流程的兴趣，为企业系统提供更快、更一致的漏洞评估。现有的用于渗透测试的LLM代理主要依赖于自我引导推理，这可能会产生不准确或幻觉的程序步骤。因此，LLM代理可能会采取非生产性的行动，例如利用未使用的软件库或生成重复先前策略的周期性响应。在这项工作中，我们提出了一个用于渗透测试LLM代理的引导推理管道，该管道结合了从MITRE ATT & CK矩阵（一个经过验证的渗透测试kll链）构建的确定性任务树，以将LLM的重组过程限制为明确定义的策略、技术和程序。这将推理锚定在经过验证的渗透测试方法中，并通过引导代理走向更有成效的攻击程序来过滤无效操作。为了评估我们的方法，我们使用三个LLM（Llama-3-8B、Gemini-1.5和GPT-4）构建了一个自动渗透测试LLM代理，并将其应用于导航10个HackTheBox网络安全演习，其中包含103个代表现实世界网络攻击场景的离散子任务。我们提出的推理管道使用Llama-3-8B、Gemini-1.5和GPT-4分别引导LLM代理完成71.8%、72.8%和78.6%的子任务。相比之下，使用自我引导推理的最先进的LLM渗透测试工具仅完成了13.5%、16.5%和75.7%的子任务，并且需要86.2%、118.7%和205.9%的模型查询。这表明将确定性任务树纳入LLM推理管道可以提高自动化网络安全评估的准确性和效率



## **3. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

大型多模式模型的鲁棒适应用于检索增强仇恨模因检测 cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2502.13061v3) [paper-pdf](http://arxiv.org/pdf/2502.13061v3)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL

摘要: 仇恨模因已成为互联网上的一个重要问题，需要强大的自动化检测系统。虽然大型多模式模型（LSYS）在仇恨模因检测方面表现出了希望，但它们面临着显着的挑战，例如次优的性能和有限的域外概括能力。最近的研究进一步揭示了在这种环境下将监督微调（SFT）和上下文学习应用于LSYS时的局限性。为了解决这些问题，我们提出了一个用于仇恨模因检测的鲁棒适应框架，该框架可以增强领域内准确性和跨领域概括性，同时保留Letts的一般视觉语言能力。分析表明，与SFT模型相比，我们的方法在对抗攻击下实现了更好的鲁棒性。对六个模因分类数据集的实验表明，我们的方法实现了最先进的性能，优于更大的代理系统。此外，与标准SFT相比，我们的方法为解释仇恨内容生成了更高质量的理由，增强了模型的可解释性。代码可访问https://github.com/JingbiaoMei/RGCL



## **4. AgentSentinel: An End-to-End and Real-Time Security Defense Framework for Computer-Use Agents**

AgentSentinel：计算机使用代理的端到端实时安全防御框架 cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07764v1) [paper-pdf](http://arxiv.org/pdf/2509.07764v1)

**Authors**: Haitao Hu, Peng Chen, Yanpeng Zhao, Yuqi Chen

**Abstract**: Large Language Models (LLMs) have been increasingly integrated into computer-use agents, which can autonomously operate tools on a user's computer to accomplish complex tasks. However, due to the inherently unstable and unpredictable nature of LLM outputs, they may issue unintended tool commands or incorrect inputs, leading to potentially harmful operations. Unlike traditional security risks stemming from insecure user prompts, tool execution results from LLM-driven decisions introduce new and unique security challenges. These vulnerabilities span across all components of a computer-use agent. To mitigate these risks, we propose AgentSentinel, an end-to-end, real-time defense framework designed to mitigate potential security threats on a user's computer. AgentSentinel intercepts all sensitive operations within agent-related services and halts execution until a comprehensive security audit is completed. Our security auditing mechanism introduces a novel inspection process that correlates the current task context with system traces generated during task execution. To thoroughly evaluate AgentSentinel, we present BadComputerUse, a benchmark consisting of 60 diverse attack scenarios across six attack categories. The benchmark demonstrates a 87% average attack success rate on four state-of-the-art LLMs. Our evaluation shows that AgentSentinel achieves an average defense success rate of 79.6%, significantly outperforming all baseline defenses.

摘要: 大型语言模型（LLM）已越来越多地集成到计算机使用代理中，这些代理可以在用户计算机上自主操作工具来完成复杂的任务。然而，由于LLM输出本质上不稳定和不可预测，它们可能会发出意外的工具命令或不正确的输入，从而导致潜在的有害操作。与由不安全的用户提示产生的传统安全风险不同，LLM驱动的决策产生的工具执行会带来新的独特安全挑战。这些漏洞跨越计算机使用代理的所有组件。为了减轻这些风险，我们提出了AgentSentinel，这是一种端到端的实时防御框架，旨在减轻用户计算机上的潜在安全威胁。AgentSentinel拦截代理相关服务中的所有敏感操作，并停止执行，直到完成全面的安全审计。我们的安全审计机制引入了一种新颖的检查过程，该过程将当前任务上下文与任务执行期间生成的系统跟踪关联起来。为了彻底评估AgentSentinel，我们介绍了BadComputerUse，这是一个由六种攻击类别的60种不同攻击场景组成的基准。该基准显示，四种最先进的LLM的平均攻击成功率为87%。我们的评估显示，AgentSentinel的平均防御成功率为79.6%，显着优于所有基线防御。



## **5. TrojanRobot: Physical-world Backdoor Attacks Against VLM-based Robotic Manipulation**

TrojanRobot：针对基于VLM的机器人操纵的物理世界后门攻击 cs.RO

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2411.11683v4) [paper-pdf](http://arxiv.org/pdf/2411.11683v4)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Aishan Liu, Leo Yu Zhang, Xiaohua Jia

**Abstract**: Robotic manipulation in the physical world is increasingly empowered by \textit{large language models} (LLMs) and \textit{vision-language models} (VLMs), leveraging their understanding and perception capabilities. Recently, various attacks against such robotic policies have been proposed, with backdoor attacks drawing considerable attention for their high stealth and strong persistence capabilities. However, existing backdoor efforts are limited to simulators and suffer from physical-world realization. To address this, we propose \textit{TrojanRobot}, a highly stealthy and broadly effective robotic backdoor attack in the physical world. Specifically, we introduce a module-poisoning approach by embedding a backdoor module into the modular robotic policy, enabling backdoor control over the policy's visual perception module thereby backdooring the entire robotic policy. Our vanilla implementation leverages a backdoor-finetuned VLM to serve as the backdoor module. To enhance its generalization in physical environments, we propose a prime implementation, leveraging the LVLM-as-a-backdoor paradigm and developing three types of prime attacks, \ie, \textit{permutation}, \textit{stagnation}, and \textit{intentional} attacks, thus achieving finer-grained backdoors. Extensive experiments on the UR3e manipulator with 18 task instructions using robotic policies based on four VLMs demonstrate the broad effectiveness and physical-world stealth of TrojanRobot. Our attack's video demonstrations are available via a github link https://trojanrobot.github.io.

摘要: \textit{大型语言模型}（LLM）和\textit{视觉语言模型}（VLMS）利用它们的理解和感知能力，越来越多地增强物理世界中的机器人操纵能力。最近，针对此类机器人策略的各种攻击被提出，其中后门攻击因其高隐身性和强持久性能力而引起了相当大的关注。然而，现有的后门工作仅限于模拟器，并且受到物理世界实现的影响。为了解决这个问题，我们提出了\textit{TrojanRobot}，这是物理世界中一种高度隐蔽且广泛有效的机器人后门攻击。具体来说，我们通过将后门模块嵌入模块到模块化机器人策略中来引入模块中毒方法，从而对策略的视觉感知模块进行后门控制，从而后门化整个机器人策略。我们的普通实现利用一个经过后门微调的VLM作为后门模块。为了增强其在物理环境中的通用性，我们提出了一种主要实现，利用LVLM作为后门范式并开发三种类型的主要攻击，即\textit{perspective}、\textit{staduction}和\textit{intentional}攻击，从而实现更细粒度的后门。UR 3e机械手与18个任务指令使用机器人策略的基础上，四个VLMs的广泛实验证明了广泛的有效性和物理世界的隐身TrojanRobot。我们的攻击视频演示可以通过github链接https://trojanrobot.github.io获得。



## **6. Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling**

通过激活引导MCMC采样的可转移直接即时注射 cs.AI

Accepted to EMNLP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07617v1) [paper-pdf](http://arxiv.org/pdf/2509.07617v1)

**Authors**: Minghui Li, Hao Zhang, Yechao Zhang, Wei Wan, Shengshan Hu, pei Xiaobing, Jing Wang

**Abstract**: Direct Prompt Injection (DPI) attacks pose a critical security threat to Large Language Models (LLMs) due to their low barrier of execution and high potential damage. To address the impracticality of existing white-box/gray-box methods and the poor transferability of black-box methods, we propose an activations-guided prompt injection attack framework. We first construct an Energy-based Model (EBM) using activations from a surrogate model to evaluate the quality of adversarial prompts. Guided by the trained EBM, we employ the token-level Markov Chain Monte Carlo (MCMC) sampling to adaptively optimize adversarial prompts, thereby enabling gradient-free black-box attacks. Experimental results demonstrate our superior cross-model transferability, achieving 49.6% attack success rate (ASR) across five mainstream LLMs and 34.6% improvement over human-crafted prompts, and maintaining 36.6% ASR on unseen task scenarios. Interpretability analysis reveals a correlation between activations and attack effectiveness, highlighting the critical role of semantic patterns in transferable vulnerability exploitation.

摘要: 直接提示注入（DPI）攻击由于其低执行门槛和高潜在危害性，对大型语言模型（LLM）构成了严重的安全威胁。针对现有白盒/灰盒方法的不实用性和黑盒方法的可移植性差的问题，提出了一种激活引导的提示注入攻击框架.首先，我们构建了一个基于能量的模型（EBM）使用代理模型的激活来评估对抗性提示的质量。在经过训练的EBM的指导下，我们采用令牌级马尔可夫链蒙特卡罗（MCMC）采样来自适应地优化对抗性提示，从而实现无梯度的黑盒攻击。实验结果证明了我们卓越的跨模型可移植性，在五种主流LLM中实现了49.6%的攻击成功率（ASB），比人工制作的提示提高了34.6%，并在未见的任务场景中保持了36.6%的ASB。可解释性分析揭示了激活和攻击有效性之间的相关性，凸显了语义模式在可转移漏洞利用中的关键作用。



## **7. A Decade-long Landscape of Advanced Persistent Threats: Longitudinal Analysis and Global Trends**

长达十年的高级持续威胁格局：纵向分析和全球趋势 cs.CR

18 pages, 13 figures (including subfigures), 11 tables. To appear in  the Proceedings of the ACM Conference on Computer and Communications Security  (CCS) 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07457v1) [paper-pdf](http://arxiv.org/pdf/2509.07457v1)

**Authors**: Shakhzod Yuldoshkhujaev, Mijin Jeon, Doowon Kim, Nick Nikiforakis, Hyungjoon Koo

**Abstract**: An advanced persistent threat (APT) refers to a covert, long-term cyberattack, typically conducted by state-sponsored actors, targeting critical sectors and often remaining undetected for long periods. In response, collective intelligence from around the globe collaborates to identify and trace surreptitious activities, generating substantial documentation on APT campaigns publicly available on the web. While prior works predominantly focus on specific aspects of APT cases, such as detection, evaluation, cyber threat intelligence, and dataset creation, limited attention has been devoted to revisiting and investigating these scattered dossiers in a longitudinal manner. The objective of our study is to fill the gap by offering a macro perspective, connecting key insights and global trends in past APT attacks. We systematically analyze six reliable sources-three focused on technical reports and another three on threat actors-examining 1,509 APT dossiers (24,215 pages) spanning 2014-2023, and identifying 603 unique APT groups worldwide. To efficiently unearth relevant information, we employ a hybrid methodology that combines rule-based information retrieval with large-language-model-based search techniques. Our longitudinal analysis reveals shifts in threat actor activities, global attack vectors, changes in targeted sectors, and relationships between cyberattacks and significant events such as elections or wars, which provide insights into historical patterns in APT evolution. Over the past decade, 154 countries have been affected, primarily using malicious documents and spear phishing as dominant initial infiltration vectors, with a noticeable decline in zero-day exploitation since 2016. Furthermore, we present our findings through interactive visualization tools, such as an APT map or flow diagram, to facilitate intuitive understanding of global patterns and trends in APT activities.

摘要: 高级持续威胁（APT）是指秘密的长期网络攻击，通常由国家支持的行为者实施，针对关键部门，并且通常长时间未被发现。作为回应，来自全球各地的集体情报机构合作识别和追踪秘密活动，在网上公开生成有关APT活动的大量文档。虽然之前的工作主要集中在APT案例的特定方面，例如检测、评估、网络威胁情报和数据集创建，但人们对以纵向方式重新访问和调查这些分散的档案的注意力有限。我们研究的目标是通过提供宏观视角、联系过去APT攻击的关键见解和全球趋势来填补这一空白。我们系统地分析了六个可靠的来源，其中三个集中在技术报告上，另外三个集中在威胁行为者上，检查了2014-2023年的1，509份APT档案（24，215页），并确定了全球603个独特的APT组织。为了有效地挖掘相关信息，我们采用了一种混合的方法，结合基于规则的信息检索与基于大语言模型的搜索技术。我们的纵向分析揭示了威胁行为者活动的变化、全球攻击载体、目标行业的变化以及网络攻击与选举或战争等重大事件之间的关系，从而深入了解APT演变的历史模式。在过去的十年中，154个国家受到影响，主要使用恶意文件和鱼叉式网络钓鱼作为主要的初始渗透载体，自2016年以来零日攻击明显下降。此外，我们还通过交互式可视化工具（例如APT地图或流程图）展示我们的发现，以促进对APT活动的全球模式和趋势的直观理解。



## **8. OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models**

Obliviate：针对大型语言模型的稳健且实用的机器去学习 cs.CL

To appear at EMNLP 25 main conference

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.04416v2) [paper-pdf](http://arxiv.org/pdf/2505.04416v2)

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose \textbf{OBLIVIATE}, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA) ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: \emph{forget quality} (via a new document-level memorization score), \emph{model utility}, and \emph{fluency}. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios.

摘要: 在广泛的库中训练的大型语言模型（LLM）存在记忆敏感、受版权保护或有毒内容的风险。为了解决这个问题，我们提出了\textBF{ObLIVIATE}，这是一个强大的去学习框架，可以在保留模型效用的同时删除目标数据。该框架遵循一个结构化过程：提取目标令牌、构建保留集以及使用定制的损失函数进行微调，该函数包括三个部分--掩蔽、蒸馏和世界事实。使用低级适配器（LoRA）可以在不影响取消学习质量的情况下确保效率。我们使用一套全面的指标对多个数据集进行实验，包括《哈利·波特》系列、WMDP和TOFU：\{忘记质量}（通过新的文档级记忆评分）、\{模型实用程序}和\{流利度}。结果证明了它在抵抗隶属度推理攻击、最大限度地减少对保留数据的影响以及在不同场景下保持稳健性方面的有效性。



## **9. GRADA: Graph-based Reranking against Adversarial Documents Attack**

GRADA：基于图的重新排名对抗文档攻击 cs.IR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.07546v2) [paper-pdf](http://arxiv.org/pdf/2505.07546v2)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.

摘要: 检索增强生成（RAG）框架通过集成来自检索文档的外部知识来提高大型语言模型（LLM）的准确性，从而克服模型静态内在知识的局限性。然而，这些系统很容易受到对抗性攻击，这些攻击通过引入对抗性但在语义上与查询相似的文档来操纵检索过程。值得注意的是，虽然这些对抗性文档类似于查询，但它们与检索集中的良性文档表现出弱的相似性。因此，我们提出了一个简单而有效的基于图形的对抗性文档攻击重新排名（GRADA）框架，旨在保留检索质量，同时显着降低对手的成功。我们的研究通过在五个LLM上进行的实验来评估我们的方法的有效性：GPT-3.5-Turbo，GPT-4 o，Llama3.1-8b，Llama3.1- 70 b和Qwen2.5- 7 b。我们使用三个数据集来评估性能，来自Natural Questions数据集的结果表明攻击成功率降低了80%，同时保持了最小的准确性损失。



## **10. Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection**

多轮会话中的社会工程个性化攻击：LLM Agent仿真与检测 cs.CR

Accepted as a paper at COLM 2025 Workshop on AI Agents: Capabilities  and Safety

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.15552v2) [paper-pdf](http://arxiv.org/pdf/2503.15552v2)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the SE attack mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.

摘要: 会话代理的快速发展，特别是由大型语言模型（LLM）驱动的聊天机器人，对社交媒体平台构成了社会工程（SE）攻击的重大风险。由于这些会话的动态性质，多回合、基于聊天的交互中的SE检测比单实例检测复杂得多。减轻这种威胁的一个关键因素是了解SE攻击的机制，特别是攻击者如何利用漏洞以及受害者的个性特征如何影响他们的易感性。在这项工作中，我们提出了一个LLM-agentic框架，SE-VSim，模拟SE攻击机制，通过生成多轮对话。我们对具有不同性格特征的受害者特工进行建模，以评估心理特征如何影响操纵的易感性。我们使用包含1000多个模拟对话的数据集，检查了冒充招聘人员、资助机构和记者的对手试图提取敏感信息的攻击场景。基于此分析，我们提出了一个概念验证SE-OmniGuard，通过利用受害者个性的先验知识、评估攻击策略以及监控对话中的信息交换以识别潜在的SE尝试，为用户提供个性化保护。



## **11. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

攻击LLM和AI代理：针对大型语言模型的广告嵌入攻击 cs.CR

6 pages, 2 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2508.17674v2) [paper-pdf](http://arxiv.org/pdf/2508.17674v2)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.

摘要: 我们引入了广告嵌入攻击（AEA），这是一种新型LLM安全威胁，可以悄悄地将促销或恶意内容注入模型输出和AI代理中。AEA通过两种低成本载体运作：（1）劫持第三方服务分发平台以预先设置对抗提示，以及（2）发布经过攻击者数据微调的后门开源检查点。与降低准确性的传统攻击不同，AEA破坏了信息完整性，导致模型在看起来正常的情况下返回秘密广告、宣传或仇恨言论。我们详细介绍了攻击管道，绘制了五个利益相关者受害者群体，并提出了一种初步的基于预算的自我检查防御，该防御可以减轻这些注入，而无需额外的模型再培训。我们的调查结果揭示了LLM安全方面存在一个紧迫且未充分解决的差距，并呼吁人工智能安全界协调检测、审计和政策响应。



## **12. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

将逻辑与自身对立：通过对比问题探索模型辩护 cs.CL

Accepted at EMNLP 2025 (Main)

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2501.01872v4) [paper-pdf](http://arxiv.org/pdf/2501.01872v4)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.

摘要: 尽管大型语言模型与人类价值观和道德原则广泛一致，但仍然容易受到利用其推理能力的复杂越狱攻击。现有的安全措施通常检测到明显的恶意意图，但无法解决微妙的、推理驱动的漏洞。在这项工作中，我们引入了POATE（极反相查询生成、对抗模板构建和搜索），这是一种新颖的越狱技术，利用对比推理来引发不道德的反应。POATE精心设计了语义上相反的意图，并将它们与对抗模板集成，以非凡的微妙性引导模型走向有害的输出。我们对参数大小不同的六个不同语言模型家族进行了广泛的评估，以证明攻击的稳健性，与现有方法相比，实现了显着更高的攻击成功率（~44%）。为了解决这个问题，我们提出了意图感知CoT和反向思维CoT，它们分解查询以检测恶意意图，并反向推理以评估和拒绝有害响应。这些方法增强了推理的稳健性并加强了模型对对抗性利用的防御。



## **13. Mind Your Server: A Systematic Study of Parasitic Toolchain Attacks on the MCP Ecosystem**

注意您的服务器：对CP生态系统的寄生工具链攻击的系统研究 cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06572v1) [paper-pdf](http://arxiv.org/pdf/2509.06572v1)

**Authors**: Shuli Zhao, Qinsheng Hou, Zihan Zhan, Yanhao Wang, Yuchong Xie, Yu Guo, Libo Chen, Shenghong Li, Zhi Xue

**Abstract**: Large language models (LLMs) are increasingly integrated with external systems through the Model Context Protocol (MCP), which standardizes tool invocation and has rapidly become a backbone for LLM-powered applications. While this paradigm enhances functionality, it also introduces a fundamental security shift: LLMs transition from passive information processors to autonomous orchestrators of task-oriented toolchains, expanding the attack surface, elevating adversarial goals from manipulating single outputs to hijacking entire execution flows. In this paper, we reveal a new class of attacks, Parasitic Toolchain Attacks, instantiated as MCP Unintended Privacy Disclosure (MCP-UPD). These attacks require no direct victim interaction; instead, adversaries embed malicious instructions into external data sources that LLMs access during legitimate tasks. The malicious logic infiltrates the toolchain and unfolds in three phases: Parasitic Ingestion, Privacy Collection, and Privacy Disclosure, culminating in stealthy exfiltration of private data. Our root cause analysis reveals that MCP lacks both context-tool isolation and least-privilege enforcement, enabling adversarial instructions to propagate unchecked into sensitive tool invocations. To assess the severity, we design MCP-SEC and conduct the first large-scale security census of the MCP ecosystem, analyzing 12,230 tools across 1,360 servers. Our findings show that the MCP ecosystem is rife with exploitable gadgets and diverse attack methods, underscoring systemic risks in MCP platforms and the urgent need for defense mechanisms in LLM-integrated environments.

摘要: 大型语言模型（LLM）通过模型上下文协议（HCP）越来越多地与外部系统集成，该协议使工具调用同步化，并已迅速成为LLM支持的应用程序的支柱。虽然这种范式增强了功能，但它也引入了根本性的安全转变：LLM从被动信息处理器过渡到面向任务的工具链的自主编排，扩大了攻击面，将对抗目标从操纵单个输出提升到劫持整个执行流。在本文中，我们揭示了一类新的攻击，即寄生工具链攻击，实例化为LCP无意隐私泄露（MCP-UPD）。这些攻击不需要受害者直接互动;相反，对手会将恶意指令嵌入到LLM在合法任务期间访问的外部数据源中。恶意逻辑渗透到工具链中，并分三个阶段展开：寄生摄入、隐私收集和隐私披露，最终导致私人数据的秘密泄露。我们的根本原因分析表明，LCP缺乏上下文工具隔离和最低特权强制执行，使得对抗指令能够不受限制地传播到敏感工具调用中。为了评估严重性，我们设计了MCP-SEC，并对LCP生态系统进行了首次大规模安全普查，分析了1，360台服务器上的12，230个工具。我们的研究结果表明，LCP生态系统中充斥着可利用的小工具和多样化的攻击方法，凸显了LCP平台的系统性风险以及LLM集成环境中对防御机制的迫切需求。



## **14. Soft Token Attacks Cannot Reliably Audit Unlearning in Large Language Models**

软令牌攻击无法可靠地审计大型语言模型中的取消学习 cs.CL

EMNLP 2025 Findings

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2502.15836v2) [paper-pdf](http://arxiv.org/pdf/2502.15836v2)

**Authors**: Haokun Chen, Sebastian Szyller, Weilin Xu, Nageen Himayat

**Abstract**: Large language models (LLMs) are trained using massive datasets, which often contain undesirable content such as harmful texts, personal information, and copyrighted material. To address this, machine unlearning aims to remove information from trained models. Recent work has shown that soft token attacks (STA) can successfully extract unlearned information from LLMs, but in this work we show that STAs can be an inadequate tool for auditing unlearning. Using common benchmarks such as Who Is Harry Potter? and TOFU, we demonstrate that in a strong auditor setting such attacks can elicit any information from the LLM, regardless of the deployed unlearning algorithm or whether the queried content was originally present in the training corpus. We further show that STA with just a few soft tokens (1-10) can elicit random strings over 400 characters long, indicating that STAs must be used carefully to effectively audit unlearning. Example code can be found at: https://github.com/IntelLabs/LLMart/tree/main/examples/unlearning

摘要: 大型语言模型（LLM）使用大量数据集进行训练，这些数据集通常包含有害文本，个人信息和版权材料等不良内容。为了解决这个问题，机器学习旨在从训练模型中删除信息。最近的工作表明，软令牌攻击（STA）可以成功地从LLM中提取未学习的信息，但在这项工作中，我们表明，STA可以是一个不充分的工具，审计unlearning。使用常见的基准，如谁是哈利波特？和TOFU，我们证明，在一个强大的审计设置这样的攻击可以引发任何信息的LLM，无论部署的unlearning算法或查询的内容是否最初存在于训练语料库。我们进一步表明，STA只有几个软令牌（1-10）可以引发超过400个字符长的随机字符串，这表明STA必须小心使用，以有效地审计遗忘。示例代码可以在https://github.com/IntelLabs/LLMart/tree/main/examples/unlearning上找到



## **15. Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks**

您的基于LLM的文本到SQL模型安全吗？通过后门攻击探索SQL注入 cs.CR

Accepted to SIGMOD 2026

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.05445v3) [paper-pdf](http://arxiv.org/pdf/2503.05445v3)

**Authors**: Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang

**Abstract**: Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.

摘要: 大型语言模型（LLM）在将自然语言问题翻译为SQL查询（文本到SQL）方面显示出了最先进的结果，这是数据库界长期存在的挑战。然而，安全问题在很大程度上仍未得到探讨，特别是后门攻击的威胁，后门攻击可以通过对有毒数据集进行微调将恶意行为引入模型。在这项工作中，我们系统地研究了基于LLM的文本到SQL模型的漏洞，并提出了ToxicSQL，这是一种新型后门攻击框架。我们的方法利用隐形的（语义和字符级触发器）来使后门难以检测和删除，确保恶意行为保持隐蔽，同时对良性输入保持高模型准确性。此外，我们建议利用SQL注入有效负载作为后门目标，从而生成恶意但可执行的SQL查询，这在基于语言模型的SQL开发中构成了严重的安全和隐私风险。我们证明，仅注入0.44%的有毒数据就会导致79.41%的攻击成功率，对数据库安全构成重大风险。此外，我们还提出了检测和缓解策略来增强模型的可靠性。我们的研究结果强调了对安全意识的文本到SQL开发的迫切需求，并强调了针对后门威胁的强大防御的重要性。



## **16. Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?**

Mask-GCG：敌对后缀中的所有代币都是越狱攻击所必需的吗？ cs.CL

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06350v1) [paper-pdf](http://arxiv.org/pdf/2509.06350v1)

**Authors**: Junjie Mu, Zonghao Ying, Zhekui Fan, Zonglei Jing, Yaoyuan Zhang, Zhengmin Yu, Wenxin Zhang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks.

摘要: 对大型语言模型（LLM）的越狱攻击已经展示了各种成功的方法，攻击者通过这些方法操纵模型来生成他们旨在避免的有害响应。其中，贪婪坐标梯度（GCG）已成为一种通用且有效的方法，可以优化后缀中的标记以生成可越狱的提示。虽然已经提出了GCG的几种改进变体，但它们都依赖于固定长度的后缀。然而，这些后缀中的潜在冗余仍有待探索。在这项工作中，我们提出了Mask-GCG，这是一种即插即用方法，它采用可学习的标记掩蔽来识别后缀内有影响力的标记。我们的方法增加了高影响力位置上代币的更新概率，同时修剪低影响力位置上的代币。与GCG相比，这种修剪不仅减少了冗余，还减少了梯度空间的大小，从而降低了计算负担并缩短了实现成功攻击所需的时间。我们通过将Mask-GCG应用于原始GCG和几种改进的变体来评估Mask-GCG。实验结果表明，后缀中的大多数令牌对攻击成功做出了显着贡献，并且修剪少数低影响令牌不会影响损失值或损害攻击成功率（ASB），从而揭示了LLM提示中的令牌冗余。我们的研究结果为从越狱攻击的角度开发高效且可解释的LLM提供了见解。



## **17. Embedding Poisoning: Bypassing Safety Alignment via Embedding Semantic Shift**

嵌入中毒：通过嵌入语义转变来实现安全一致 cs.CR

16 pages,9 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06338v1) [paper-pdf](http://arxiv.org/pdf/2509.06338v1)

**Authors**: Shuai Yuan, Zhibo Zhang, Yuxi Li, Guangdong Bai, Wang Kailong

**Abstract**: The widespread distribution of Large Language Models (LLMs) through public platforms like Hugging Face introduces significant security challenges. While these platforms perform basic security scans, they often fail to detect subtle manipulations within the embedding layer. This work identifies a novel class of deployment phase attacks that exploit this vulnerability by injecting imperceptible perturbations directly into the embedding layer outputs without modifying model weights or input text. These perturbations, though statistically benign, systematically bypass safety alignment mechanisms and induce harmful behaviors during inference. We propose Search based Embedding Poisoning(SEP), a practical, model agnostic framework that introduces carefully optimized perturbations into embeddings associated with high risk tokens. SEP leverages a predictable linear transition in model responses, from refusal to harmful output to semantic deviation to identify a narrow perturbation window that evades alignment safeguards. Evaluated across six aligned LLMs, SEP achieves an average attack success rate of 96.43% while preserving benign task performance and evading conventional detection mechanisms. Our findings reveal a critical oversight in deployment security and emphasize the urgent need for embedding level integrity checks in future LLM defense strategies.

摘要: 通过Hugging Face等公共平台广泛分发大型语言模型（LLM）带来了重大的安全挑战。虽然这些平台执行基本的安全扫描，但它们通常无法检测到嵌入层内的微妙操纵。这项工作识别了一类新型的部署阶段攻击，这些攻击通过在不修改模型权重或输入文本的情况下直接将不可感知的扰动注入到嵌入层输出中来利用该漏洞。这些扰动虽然在统计上是良性的，但系统性地绕过了安全对齐机制，并在推理过程中引发有害行为。我们提出了基于搜索的嵌入中毒（SDP），这是一个实用的、模型不可知的框架，它将精心优化的扰动引入到与高风险代币相关的嵌入中。BEP利用模型响应中的可预测线性转变（从拒绝到有害输出再到语义偏差）来识别逃避对齐保障措施的狭窄扰动窗口。通过对六个对齐的LLM进行评估，SDP的平均攻击成功率为96.43%，同时保持良性任务性能并规避传统检测机制。我们的调查结果揭示了部署安全方面的严重疏忽，并强调迫切需要在未来的LLM防御策略中嵌入级别完整性检查。



## **18. AttestLLM: Efficient Attestation Framework for Billion-scale On-device LLMs**

AttestLLM：用于数十亿规模设备上LLM的高效认证框架 cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06326v1) [paper-pdf](http://arxiv.org/pdf/2509.06326v1)

**Authors**: Ruisi Zhang, Yifei Zhao, Neusha Javidnia, Mengxin Zheng, Farinaz Koushanfar

**Abstract**: As on-device LLMs(e.g., Apple on-device Intelligence) are widely adopted to reduce network dependency, improve privacy, and enhance responsiveness, verifying the legitimacy of models running on local devices becomes critical. Existing attestation techniques are not suitable for billion-parameter Large Language Models (LLMs), struggling to remain both time- and memory-efficient while addressing emerging threats in the LLM era. In this paper, we present AttestLLM, the first-of-its-kind attestation framework to protect the hardware-level intellectual property (IP) of device vendors by ensuring that only authorized LLMs can execute on target platforms. AttestLLM leverages an algorithm/software/hardware co-design approach to embed robust watermarking signatures onto the activation distributions of LLM building blocks. It also optimizes the attestation protocol within the Trusted Execution Environment (TEE), providing efficient verification without compromising inference throughput. Extensive proof-of-concept evaluations on LLMs from Llama, Qwen, and Phi families for on-device use cases demonstrate AttestLLM's attestation reliability, fidelity, and efficiency. Furthermore, AttestLLM enforces model legitimacy and exhibits resilience against model replacement and forgery attacks.

摘要: 作为设备上LLM（例如，Apple设备上智能）被广泛采用以减少网络依赖性、改善隐私和增强响应能力，验证在本地设备上运行的型号的合法性变得至关重要。现有的证明技术不适合数十亿参数的大型语言模型（LLM），在解决LLM时代新出现的威胁时，难以保持时间和内存效率。在本文中，我们介绍了AttestLLM，这是首个同类认证框架，旨在通过确保只有授权的LLM才能在目标平台上执行来保护设备供应商的硬件级知识产权（IP）。AttestLLM利用算法/软件/硬件协同设计方法将稳健的水印签名嵌入到LLM构建块的激活分布中。它还优化了可信执行环境（TEK）中的证明协议，在不影响推理吞吐量的情况下提供高效验证。针对设备上用例对Llama、Qwen和Phi系列的LLM进行了广泛的概念验证评估，证明了AttestLLM的证明可靠性、保真度和效率。此外，AttestLLM强制执行模型合法性，并表现出针对模型替换和伪造攻击的弹性。



## **19. Multimodal Prompt Injection Attacks: Risks and Defenses for Modern LLMs**

多模式即时注入攻击：现代LLM的风险和防御 cs.CR

8 pages, 4 figures, 2 tables

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2509.05883v1) [paper-pdf](http://arxiv.org/pdf/2509.05883v1)

**Authors**: Andrew Yeo, Daeseon Choi

**Abstract**: Large Language Models (LLMs) have seen rapid adoption in recent years, with industries increasingly relying on them to maintain a competitive advantage. These models excel at interpreting user instructions and generating human-like responses, leading to their integration across diverse domains, including consulting and information retrieval. However, their widespread deployment also introduces substantial security risks, most notably in the form of prompt injection and jailbreak attacks.   To systematically evaluate LLM vulnerabilities -- particularly to external prompt injection -- we conducted a series of experiments on eight commercial models. Each model was tested without supplementary sanitization, relying solely on its built-in safeguards. The results exposed exploitable weaknesses and emphasized the need for stronger security measures. Four categories of attacks were examined: direct injection, indirect (external) injection, image-based injection, and prompt leakage. Comparative analysis indicated that Claude 3 demonstrated relatively greater robustness; nevertheless, empirical findings confirm that additional defenses, such as input normalization, remain necessary to achieve reliable protection.

摘要: 近年来，大型语言模型（LLM）得到了迅速采用，行业越来越依赖它们来保持竞争优势。这些模型擅长解释用户指令并生成类似人类的响应，从而使它们能够在咨询和信息检索等不同领域进行集成。然而，它们的广泛部署也带来了巨大的安全风险，最引人注目的是迅速注入和越狱攻击的形式。   为了系统性地评估LLM漏洞--特别是外部即时注入--我们对八个商业模型进行了一系列实验。每个型号的测试都没有进行补充清理，仅依赖其内置的保障措施。结果暴露了可利用的弱点，并强调需要采取更强有力的安全措施。检查了四种类型的攻击：直接注射、间接（外部）注射、基于图像的注射和即时泄漏。比较分析表明Claude 3表现出相对更高的鲁棒性;然而，经验研究结果证实，为了实现可靠的保护，仍然需要额外的防御措施，例如输入正常化。



## **20. Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization**

解码LLM中的潜在攻击：通过Web摘要中的HTML提示注入 cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05831v1) [paper-pdf](http://arxiv.org/pdf/2509.05831v1)

**Authors**: Ishaan Verma

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.

摘要: 大型语言模型（LLM）越来越多地集成到基于Web的内容摘要系统中，但它们对即时注入攻击的敏感性仍然是一个紧迫的问题。在这项研究中，我们探索了如何利用非可见的HTML元素（例如<meta>、咏叹调标签和alt属性）来嵌入对抗性指令，而不改变网页的可见内容。我们引入了一个由280个静态网页组成的新颖数据集，平均分为干净和对抗注入版本，使用不同的基于HTML的策略制作。这些页面通过浏览器自动化管道进行处理，以提取原始HTML和渲染文本，密切模仿现实世界的LLM部署场景。我们评估了两个最先进的开源模型Llama 4 Scout（Meta）和Gemma 9 B IT（Google）总结此内容的能力。使用词汇（ROUGE-L）和语义（SBERT cos相似性）指标以及手动注释，我们评估这些隐蔽注入的影响。我们的研究结果显示，超过29%的注射样本导致Llama 4 Scout总结发生了显着变化，而Gemma 9 B IT的成功率较低，但并非微不足道，为15%。这些结果凸显了LLM驱动的网络管道中一个关键且在很大程度上被忽视的漏洞，其中隐藏的对抗内容可以巧妙地操纵模型输出。我们的工作为评估基于HTML的即时注入提供了一个可重复的框架和基准，并强调了涉及Web内容的LLM应用程序中对稳健的缓解策略的迫切需要。



## **21. Membership Inference Attacks on LLM-based Recommender Systems**

对基于LLM的推荐系统的成员推断攻击 cs.IR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2508.18665v2) [paper-pdf](http://arxiv.org/pdf/2508.18665v2)

**Authors**: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen

**Abstract**: Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.

摘要: 基于大型语言模型（LLM）的推荐系统（RecSys）可以灵活地调整推荐系统以适应不同的领域。它利用上下文学习（ICL），即提示来定制推荐功能，其中包括敏感的历史用户特定项目交互，例如，隐性反馈，例如点击的项目或明确的产品评论。此类私人信息可能会受到新型隐私攻击。然而，对于这个重要问题还没有进行任何研究。我们设计了四种成员资格推断攻击（MIA），旨在揭示受害者的历史互动是否被系统提示使用。它们是\r {直接询问、幻觉、相似性和中毒攻击}，每一种都利用了LLM或RecSys的独特功能。我们在三个用于开发ICL-LLM RecSys的LLM和两个著名的RecSys基准数据集上仔细评估了它们。结果证实，LLM RecSys上的MIA威胁是现实的：直接询问和中毒攻击显示出显着较高的攻击优势。我们还分析了影响这些攻击的因素，例如系统提示中的射击次数以及受害者在射击中的位置。



## **22. Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System**

在基于LLM的开发系统中利用工具调用提示实现工具行为劫持 cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05755v1) [paper-pdf](http://arxiv.org/pdf/2509.05755v1)

**Authors**: Yu Liu, Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.

摘要: 基于法学硕士的代理系统利用大型语言模型来处理用户查询、做出决策并执行外部工具，以执行跨聊天机器人、客户服务和软件工程等领域的复杂任务。这些系统的一个关键组件是工具调用提示（TIP），它定义了工具交互协议并指导LLM确保工具使用的安全性和正确性。尽管TIP的安全性很重要，但在很大程度上被忽视了。这项工作调查了与TIP相关的安全风险，揭示了Cursor、Claude Code等基于LLM的主要系统容易受到远程代码执行（RCE）和拒绝服务（NOS）等攻击。通过系统性TIP利用工作流程（TEW），我们通过操纵工具调用演示了外部工具行为劫持。我们还提出了防御机制来增强基于LLM的代理系统中的TIP安全性。



## **23. Reasoning Introduces New Poisoning Attacks Yet Makes Them More Complicated**

推理引入了新的中毒攻击，但使它们变得更加复杂 cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05739v1) [paper-pdf](http://arxiv.org/pdf/2509.05739v1)

**Authors**: Hanna Foerster, Ilia Shumailov, Yiren Zhao, Harsh Chaudhari, Jamie Hayes, Robert Mullins, Yarin Gal

**Abstract**: Early research into data poisoning attacks against Large Language Models (LLMs) demonstrated the ease with which backdoors could be injected. More recent LLMs add step-by-step reasoning, expanding the attack surface to include the intermediate chain-of-thought (CoT) and its inherent trait of decomposing problems into subproblems. Using these vectors for more stealthy poisoning, we introduce ``decomposed reasoning poison'', in which the attacker modifies only the reasoning path, leaving prompts and final answers clean, and splits the trigger across multiple, individually harmless components.   Fascinatingly, while it remains possible to inject these decomposed poisons, reliably activating them to change final answers (rather than just the CoT) is surprisingly difficult. This difficulty arises because the models can often recover from backdoors that are activated within their thought processes. Ultimately, it appears that an emergent form of backdoor robustness is originating from the reasoning capabilities of these advanced LLMs, as well as from the architectural separation between reasoning and final answer generation.

摘要: 对针对大型语言模型（LLM）的数据中毒攻击的早期研究证明了注入后门的容易性。最近的LLM添加了逐步推理，扩展了攻击面，以包括中间思想链（CoT）及其将问题分解为子问题的固有特征。使用这些载体进行更隐蔽的中毒，我们引入了“分解推理毒药”，其中攻击者仅修改推理路径，保留提示和最终答案干净，并将触发器拆分为多个单独无害的组件。   令人着迷的是，虽然仍然可以注射这些分解的毒药，但可靠地激活它们以改变最终答案（而不仅仅是CoT）却出奇地困难。出现这种困难是因为模型通常可以从其思维过程中激活的后门中恢复过来。最终，后门稳健性的一种新兴形式似乎源于这些高级LLM的推理能力，以及推理和最终答案生成之间的架构分离。



## **24. BadPromptFL: A Novel Backdoor Threat to Prompt-based Federated Learning in Multimodal Models**

BadminutFL：对多模式模型中基于预算的联邦学习的新型后门威胁 cs.LG

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2508.08040v3) [paper-pdf](http://arxiv.org/pdf/2508.08040v3)

**Authors**: Maozhen Zhang, Mengnan Zhao, Wei Wang, Bo Wang

**Abstract**: Prompt-based tuning has emerged as a lightweight alternative to full fine-tuning in large vision-language models, enabling efficient adaptation via learned contextual prompts. This paradigm has recently been extended to federated learning settings (e.g., PromptFL), where clients collaboratively train prompts under data privacy constraints. However, the security implications of prompt-based aggregation in federated multimodal learning remain largely unexplored, leaving a critical attack surface unaddressed. In this paper, we introduce \textbf{BadPromptFL}, the first backdoor attack targeting prompt-based federated learning in multimodal contrastive models. In BadPromptFL, compromised clients jointly optimize local backdoor triggers and prompt embeddings, injecting poisoned prompts into the global aggregation process. These prompts are then propagated to benign clients, enabling universal backdoor activation at inference without modifying model parameters. Leveraging the contextual learning behavior of CLIP-style architectures, BadPromptFL achieves high attack success rates (e.g., \(>90\%\)) with minimal visibility and limited client participation. Extensive experiments across multiple datasets and aggregation protocols validate the effectiveness, stealth, and generalizability of our attack, raising critical concerns about the robustness of prompt-based federated learning in real-world deployments.

摘要: 基于预算的调优已成为大型视觉语言模型中完全微调的轻量级替代方案，可以通过学习的上下文提示进行高效调整。该范式最近已扩展到联邦学习环境（例如，AtlantFL），客户在数据隐私限制下协作训练提示。然而，联邦多模式学习中基于预算的聚合的安全影响在很大程度上仍未得到探索，导致关键的攻击表面尚未得到解决。本文中，我们介绍了\textBF{BadoutFL}，这是第一个针对多模式对比模型中基于预算的联邦学习的后门攻击。在BadoutFL中，受影响的客户端联合优化本地后门触发器和提示嵌入，将有毒提示注入到全球聚合流程中。然后，这些提示被传播到良性客户端，从而在推理时启用通用后门，而无需修改模型参数。利用CLIP风格架构的上下文学习行为，BadminutFL实现了高攻击成功率（例如，\（>90\%\））可见性最低且客户参与有限。跨多个数据集和聚合协议的广泛实验验证了我们攻击的有效性、隐蔽性和可推广性，引发了人们对现实世界部署中基于预算的联邦学习稳健性的严重担忧。



## **25. Evo-MARL: Co-Evolutionary Multi-Agent Reinforcement Learning for Internalized Safety**

Evo-MARL：用于内部化安全的协同进化多智能体强化学习 cs.AI

accepted by the Trustworthy FMs workshop in ICCV 2025

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2508.03864v2) [paper-pdf](http://arxiv.org/pdf/2508.03864v2)

**Authors**: Zhenyu Pan, Yiting Zhang, Yutong Zhang, Jianshu Zhang, Haozheng Luo, Yuwei Han, Dennis Wu, Hong-Yu Chen, Philip S. Yu, Manling Li, Han Liu

**Abstract**: Multi-agent systems (MAS) built on multimodal large language models exhibit strong collaboration and performance. However, their growing openness and interaction complexity pose serious risks, notably jailbreak and adversarial attacks. Existing defenses typically rely on external guard modules, such as dedicated safety agents, to handle unsafe behaviors. Unfortunately, this paradigm faces two challenges: (1) standalone agents offer limited protection, and (2) their independence leads to single-point failure-if compromised, system-wide safety collapses. Naively increasing the number of guard agents further raises cost and complexity. To address these challenges, we propose Evo-MARL, a novel multi-agent reinforcement learning (MARL) framework that enables all task agents to jointly acquire defensive capabilities. Rather than relying on external safety modules, Evo-MARL trains each agent to simultaneously perform its primary function and resist adversarial threats, ensuring robustness without increasing system overhead or single-node failure. Furthermore, Evo-MARL integrates evolutionary search with parameter-sharing reinforcement learning to co-evolve attackers and defenders. This adversarial training paradigm internalizes safety mechanisms and continually enhances MAS performance under co-evolving threats. Experiments show that Evo-MARL reduces attack success rates by up to 22% while boosting accuracy by up to 5% on reasoning tasks-demonstrating that safety and utility can be jointly improved.

摘要: 基于多模式大型语言模型构建的多智能体系统（MAS）展现出强大的协作和性能。然而，它们日益增长的开放性和交互复杂性带来了严重的风险，特别是越狱和对抗性攻击。现有的防御系统通常依赖外部防护模块（例如专用安全代理）来处理不安全行为。不幸的是，这个范式面临着两个挑战：（1）独立代理提供的保护有限，（2）它们的独立性会导致单点故障--如果受到损害，系统范围的安全就会崩溃。天真地增加警卫特工的数量进一步增加了成本和复杂性。为了应对这些挑战，我们提出了Evo-MARL，这是一种新型的多智能体强化学习（MARL）框架，使所有任务智能体能够联合获得防御能力。Evo-MARL不是依赖外部安全模块，而是训练每个代理同时执行其主要功能并抵抗对抗威胁，在不增加系统负担或单节点故障的情况下确保稳健性。此外，Evo-MARL将进化搜索与参数共享强化学习集成，以共同进化攻击者和防御者。这种对抗性训练范式内化了安全机制，并在共同演变的威胁下不断增强MAS性能。实验表明，Evo-MARL将攻击成功率降低高达22%，同时将推理任务的准确性提高高达5%，证明安全性和实用性可以共同提高。



## **26. Behind the Mask: Benchmarking Camouflaged Jailbreaks in Large Language Models**

面具背后：大型语言模型中伪装越狱的基准 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05471v1) [paper-pdf](http://arxiv.org/pdf/2509.05471v1)

**Authors**: Youjia Zheng, Mohammad Zandsalimy, Shanu Sushmita

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to a sophisticated form of adversarial prompting known as camouflaged jailbreaking. This method embeds malicious intent within seemingly benign language to evade existing safety mechanisms. Unlike overt attacks, these subtle prompts exploit contextual ambiguity and the flexible nature of language, posing significant challenges to current defense systems. This paper investigates the construction and impact of camouflaged jailbreak prompts, emphasizing their deceptive characteristics and the limitations of traditional keyword-based detection methods. We introduce a novel benchmark dataset, Camouflaged Jailbreak Prompts, containing 500 curated examples (400 harmful and 100 benign prompts) designed to rigorously stress-test LLM safety protocols. In addition, we propose a multi-faceted evaluation framework that measures harmfulness across seven dimensions: Safety Awareness, Technical Feasibility, Implementation Safeguards, Harmful Potential, Educational Value, Content Quality, and Compliance Score. Our findings reveal a stark contrast in LLM behavior: while models demonstrate high safety and content quality with benign inputs, they exhibit a significant decline in performance and safety when confronted with camouflaged jailbreak attempts. This disparity underscores a pervasive vulnerability, highlighting the urgent need for more nuanced and adaptive security strategies to ensure the responsible and robust deployment of LLMs in real-world applications.

摘要: 大型语言模型（LLM）越来越容易受到一种复杂形式的对抗激励，即所谓的虚拟越狱。这种方法将恶意意图嵌入看似良性的语言中，以逃避现有的安全机制。与公开攻击不同，这些微妙的提示利用了上下文模糊性和语言的灵活性，对当前的防御系统构成了重大挑战。本文研究了伪装越狱提示的结构和影响，强调了其欺骗性特征和传统基于关键词的检测方法的局限性。我们引入了一个新颖的基准数据集，即伪装的越狱提示，其中包含500个精心策划的示例（400个有害提示和100个良性提示），旨在对LLM安全协议进行严格的压力测试。此外，我们还提出了一个多方面的评估框架，从七个方面衡量危害性：安全意识、技术可行性、实施保障措施、潜在危害性、教育价值、内容质量和合规评分。我们的研究结果揭示了LLM行为的鲜明对比：虽然模型在良性输入下表现出高安全性和内容质量，但当面对伪装的越狱尝试时，它们的性能和安全性却显着下降。这种差异凸显了普遍存在的漏洞，凸显了迫切需要更加细致入微和适应性的安全策略，以确保在现实世界应用程序中负责任且稳健地部署LLM。



## **27. Neural Breadcrumbs: Membership Inference Attacks on LLMs Through Hidden State and Attention Pattern Analysis**

神经面包屑：通过隐藏状态和注意力模式分析对LLM的成员推理攻击 cs.LG

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05449v1) [paper-pdf](http://arxiv.org/pdf/2509.05449v1)

**Authors**: Disha Makhija, Manoj Ghuhan Arivazhagan, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah

**Abstract**: Membership inference attacks (MIAs) reveal whether specific data was used to train machine learning models, serving as important tools for privacy auditing and compliance assessment. Recent studies have reported that MIAs perform only marginally better than random guessing against large language models, suggesting that modern pre-training approaches with massive datasets may be free from privacy leakage risks. Our work offers a complementary perspective to these findings by exploring how examining LLMs' internal representations, rather than just their outputs, may provide additional insights into potential membership inference signals. Our framework, \emph{memTrace}, follows what we call \enquote{neural breadcrumbs} extracting informative signals from transformer hidden states and attention patterns as they process candidate sequences. By analyzing layer-wise representation dynamics, attention distribution characteristics, and cross-layer transition patterns, we detect potential memorization fingerprints that traditional loss-based approaches may not capture. This approach yields strong membership detection across several model families achieving average AUC scores of 0.85 on popular MIA benchmarks. Our findings suggest that internal model behaviors can reveal aspects of training data exposure even when output-based signals appear protected, highlighting the need for further research into membership privacy and the development of more robust privacy-preserving training techniques for large language models.

摘要: 成员资格推理攻击（MIA）揭示了特定数据是否用于训练机器学习模型，作为隐私审计和合规评估的重要工具。最近的研究报告称，MIA的表现仅比针对大型语言模型的随机猜测好一点点，这表明具有大量数据集的现代预训练方法可能没有隐私泄露风险。我们的工作通过探索如何检查LLM的内部表示而不仅仅是它们的输出来为这些发现提供了补充的视角。我们的框架，memTrace，遵循我们称之为神经面包屑的方法，在处理候选序列时，从Transformer隐藏状态和注意力模式中提取信息信号。通过分析逐层表示动态、注意力分布特征和跨层转变模式，我们检测传统基于损失的方法可能无法捕获的潜在记忆指纹。这种方法在多个模型系列中产生了强大的成员身份检测，在流行的MIA基准上实现了0.85的平均AUT分数。我们的研究结果表明，即使基于输出的信号看起来受到保护，内部模型行为也可以揭示训练数据暴露的各个方面，这凸显了对成员隐私进行进一步研究以及为大型语言模型开发更强大的隐私保护训练技术的必要性。



## **28. AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection**

AgentArmor：对Agent DeliverTrace执行程序分析以防止即时注入 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2508.01249v2) [paper-pdf](http://arxiv.org/pdf/2508.01249v2)

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu

**Abstract**: Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's runtime traces as graph-based intermediate representations with control and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools \& data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis for sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can reduce the ASR to 3\%, with the utility drop only 1\%.

摘要: 大型语言模型（LLM）代理通过将自然语言推理与外部工具的执行相结合，提供了一个强大的新范式来解决各种问题。然而，它们的动态和不透明行为会带来严重的安全风险，特别是在存在即时注入攻击的情况下。在这项工作中，我们提出了一种新颖的见解，将代理运行时跟踪视为具有可分析语义的结构化程序。因此，我们提出了AgentArmor，这是一个程序分析框架，它将代理跟踪转换为基于图形中间表示的结构化程序依赖性表示（例如，CGM、DFG和PDG）并通过类型系统强制执行安全策略。AgentArmor由三个关键组件组成：（1）一个图形构造器，它将代理的运行时跟踪重建为基于图形的中间表示，其中描述了控制和数据流;（2）一个属性注册表，它附加了交互工具\&数据的安全相关元数据，以及（3）一个类型系统，它执行静态推理和检查中间表示。通过将代理行为表示为结构化程序，AgentArmor可以对敏感数据流、信任边界和策略违规进行程序分析。在AgentDojo基准测试中对AgentArmor进行了测试，结果表明，AgentArmor可以将ASR降低到3%，而效用下降仅为1%。



## **29. RINSER: Accurate API Prediction Using Masked Language Models**

RINser：使用掩蔽语言模型进行准确的API预测 cs.CY

16 pages, 8 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04887v1) [paper-pdf](http://arxiv.org/pdf/2509.04887v1)

**Authors**: Muhammad Ejaz Ahmed, Christopher Cody, Muhammad Ikram, Sean Lamont, Alsharif Abuadbba, Seyit Camtepe, Surya Nepal, Muhammad Ali Kaafar

**Abstract**: Malware authors commonly use obfuscation to hide API identities in binary files, making analysis difficult and time-consuming for a human expert to understand the behavior and intent of the program. Automatic API prediction tools are necessary to efficiently analyze unknown binaries, facilitating rapid malware triage while reducing the workload on human analysts. In this paper, we present RINSER (AccuRate API predictioN using maSked languagE model leaRning), an automated framework for predicting Windows API (WinAPI) function names. RINSER introduces the novel concept of API codeprints, a set of API-relevant assembly instructions, and supports x86 PE binaries. RINSER relies on BERT's masked language model (LM) to predict API names at scale, achieving 85.77% accuracy for normal binaries and 82.88% accuracy for stripped binaries. We evaluate RINSER on a large dataset of 4.7M API codeprints from 11,098 malware binaries, covering 4,123 unique Windows APIs, making it the largest publicly available dataset of this type. RINSER successfully discovered 65 obfuscated Windows APIs related to C2 communication, spying, and evasion in our dataset, which the commercial disassembler IDA failed to identify. Furthermore, we compared RINSER against three state-of-the-art approaches, showing over 20% higher prediction accuracy. We also demonstrated RINSER's resilience to adversarial attacks, including instruction randomization and code displacement, with a performance drop of no more than 3%.

摘要: 恶意软件作者通常使用混淆将API身份隐藏在二进制文件中，这使得人类专家理解程序的行为和意图变得困难且耗时。自动API预测工具对于有效分析未知二进制文件来说是必要的，可以促进快速恶意软件分类，同时减少人类分析师的工作量。在本文中，我们介绍了RINBER（使用maSked languagE模型leRning的ACATER API预测），这是一个用于预测Windows API（WinAPI）函数名称的自动化框架。RINser引入了API代码印的新颖概念，即一组与API相关的汇编指令，并支持x86 PE二进制文件。RINser依赖BERT的掩蔽语言模型（LM）来大规模预测API名称，正常二进制文件的准确性达到85.77%，剥离二进制文件的准确性达到82.88%。我们在一个包含来自11，098个恶意软件二进制文件的470万个API代码的大型数据集上评估了RINser，涵盖了4，123个独特的Windows API，使其成为此类类型中最大的公开可用数据集。RINBER在我们的数据集中成功发现了65个与C2通信、间谍和规避相关的模糊Windows API，但商业反汇编器IDA未能识别这些API。此外，我们将RINBER与三种最先进的方法进行了比较，结果显示预测准确性提高了20%以上。我们还展示了RINBER对对抗攻击（包括指令随机化和代码置换）的弹性，性能下降不超过3%。



## **30. Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning**

解药：微调后大型语言模型的安全调整，防止有害的微调 cs.AI

Rejected by AAAI25-AIA. Accepted by ICML25. Authors are thankful to  the anonymous reviewers from both AAAI25-AIA and ICML25

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2408.09600v3) [paper-pdf](http://arxiv.org/pdf/2408.09600v3)

**Authors**: Tiansheng Huang, Gautam Bhattacharya, Pratik Joshi, Josh Kimball, Ling Liu

**Abstract**: Safety aligned Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks -- a few harmful data mixed in the fine-tuning dataset can break the LLMs's safety alignment. While several defenses have been proposed, our evaluation shows that existing defenses fail \textit{when some specific training hyper-parameters are chosen} -- a large learning rate or a large number of training epochs in the fine-tuning stage can easily invalidate the defense. To this end, we propose Antidote, a post-fine-tuning stage solution, which remains \textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning stage}}. Antidote relies on the philosophy that by removing the harmful parameters, the harmful model can be recovered from the harmful behaviors, regardless of how those harmful parameters are formed in the fine-tuning stage. With this philosophy, we introduce a one-shot pruning stage after harmful fine-tuning to remove the harmful weights that are responsible for the generation of harmful content. Despite its embarrassing simplicity, empirical results show that Antidote can reduce harmful score while maintaining accuracy on downstream tasks. Code is available at https://github.com/git-disl/Antidote.

摘要: 安全一致的大型语言模型（LLM）容易受到有害的微调攻击--微调数据集中混合的一些有害数据可能会破坏LLM的安全一致性。虽然已经提出了几种防御措施，但我们的评估表明，现有的防御措施会失败\textit{当选择一些特定的训练超参数时} --微调阶段中的大学习率或大量训练时期很容易使防御无效。为此，我们提出了Antidote，这是一种微调后的解决方案，它仍然是\textBF{\texttit {与微调阶段中的训练超参数不可知}。解药依赖于这样的理念：通过去除有害参数，有害模型可以从有害行为中恢复，无论这些有害参数是如何在微调阶段形成的。凭借这一理念，我们在有害微调后引入一次性修剪阶段，以删除导致有害内容生成的有害权重。尽管它的简单性令人尴尬，但经验结果表明，解毒剂可以降低有害评分，同时保持下游任务的准确性。代码可在https://github.com/git-disl/Antidote上获取。



## **31. Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs**

注意差距：使用行动图评估LLM中的模型和统计级别漏洞 cs.CL

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04802v1) [paper-pdf](http://arxiv.org/pdf/2509.04802v1)

**Authors**: Ilham Wicaksono, Zekun Wu, Theo King, Adriano Koshiyama, Philip Treleaven

**Abstract**: As large language models transition to agentic systems, current safety evaluation frameworks face critical gaps in assessing deployment-specific risks. We introduce AgentSeer, an observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment. Through cross-model validation on GPT-OSS-20B and Gemini-2.0-flash using HarmBench single turn and iterative refinement attacks, we demonstrate fundamental differences between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47% ASR) versus Gemini-2.0-flash (50.00% ASR), with both models showing susceptibility to social engineering while maintaining logic-based attack resistance. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, agent transfer operations as highest-risk tools, semantic rather than syntactic vulnerability mechanisms, and context-dependent attack effectiveness, alongside model-specific security profiles in absolute ASR levels and optimal injection strategies. Direct attack transfer from model-level to agentic contexts shows degraded performance (GPT-OSS-20B: 57% human injection ASR; Gemini-2.0-flash: 28%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic evaluation gaps. These findings establish the urgent need for agentic-situation evaluation paradigms, with AgentSeer providing the standardized methodology and empirical validation.

摘要: 随着大型语言模型向代理系统过渡，当前的安全评估框架在评估特定于部署的风险方面面临着严重差距。我们引入了AgentSeer，这是一个基于可观察性的评估框架，它将代理执行分解为粒度动作和组件图，从而实现系统性代理情景评估。通过使用HarmBench单轮攻击和迭代细化攻击对GPT-OSS-20 B和Gemini-2.0-Flash进行跨模型验证，我们展示了模型级和代理级漏洞配置文件之间的根本差异。模型级评估揭示了基线差异：GPT-OSS-20 B（39.47%ASB）与Gemini-2.0-Flash（50.00%ASB），两种模型都表现出对社会工程的敏感性，同时保持基于逻辑的攻击抵抗力。然而，代理层面的评估暴露了传统评估所看不到的代理特定风险。我们发现了仅在代理环境中出现的“仅代理”漏洞，工具调用显示两种模型的ASB高出24-60%。跨模型分析揭示了通用的代理模式、作为最高风险工具的代理传输操作、语义而非语法漏洞机制、取决于上下文的攻击有效性，以及绝对ASB级别的模型特定安全配置文件和最佳注入策略。从模型级到代理上下文的直接攻击转移显示出性能下降（GPT-OSS-20 B：57%人体注射ASO; Gemini-2.0-Flash：28%），而上下文感知迭代攻击成功地破坏了在模型级失败的目标，证实了系统性评估差距。这些发现确定了对主体情境评估范式的迫切需求，AgentSeer提供了标准化的方法论和经验验证。



## **32. Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs**

突破构建：用于保护LLM的基于预算的攻击威胁模型 cs.CL

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04615v1) [paper-pdf](http://arxiv.org/pdf/2509.04615v1)

**Authors**: Brennen Hill, Surendra Parla, Venkata Abhijeeth Balabhadruni, Atharv Prajod Padmalayam, Sujay Chandra Shekara Sharma

**Abstract**: The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing.

摘要: 大型语言模型（LLM）的激增带来了关键的安全挑战，对抗行为者可以操纵输入提示造成重大伤害并规避安全一致。这些基于预算的攻击利用模型设计、培训和上下文理解中的漏洞，导致知识产权盗窃、错误信息生成和用户信任度侵蚀。系统地了解这些攻击载体是开发稳健对策的基础步骤。本文对基于预算的攻击方法进行了全面的文献调查，对它们进行了分类，以提供明确的威胁模型。通过详细介绍这些漏洞利用的机制和影响，本调查旨在为研究界构建下一代安全LLM的努力提供信息，这些LLM本质上可以抵抗未经授权的提炼、微调和编辑。



## **33. Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing**

Concept-ROT：使用模型编辑在大型语言模型中中毒概念 cs.LG

Published at ICLR 2025

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2412.13341v2) [paper-pdf](http://arxiv.org/pdf/2412.13341v2)

**Authors**: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor

**Abstract**: Model editing methods modify specific behaviors of Large Language Models by altering a small, targeted set of network weights and require very little data and compute. These methods can be used for malicious applications such as inserting misinformation or simple trojans that result in adversary-specified behaviors when a trigger word is present. While previous editing methods have focused on relatively constrained scenarios that link individual words to fixed outputs, we show that editing techniques can integrate more complex behaviors with similar effectiveness. We develop Concept-ROT, a model editing-based method that efficiently inserts trojans which not only exhibit complex output behaviors, but also trigger on high-level concepts -- presenting an entirely new class of trojan attacks. Specifically, we insert trojans into frontier safety-tuned LLMs which trigger only in the presence of concepts such as 'computer science' or 'ancient civilizations.' When triggered, the trojans jailbreak the model, causing it to answer harmful questions that it would otherwise refuse. Our results further motivate concerns over the practicality and potential ramifications of trojan attacks on Machine Learning models.

摘要: 模型编辑方法通过改变一小组有针对性的网络权重来修改大型语言模型的特定行为，并且需要很少的数据和计算。这些方法可用于恶意应用程序，例如插入错误信息或简单的特洛伊木马，当存在触发词时，这些木马会导致对手指定的行为。虽然之前的编辑方法专注于将单个单词与固定输出联系起来的相对受限的场景，但我们表明编辑技术可以以类似的效果集成更复杂的行为。我们开发了Concept-ROT，这是一种基于模型编辑的方法，可以有效地插入特洛伊木马，这些木马不仅表现出复杂的输出行为，而且还会触发高级概念--从而呈现出一种全新的特洛伊木马攻击。具体来说，我们将特洛伊木马插入到前沿安全调整的LLM中，这些LLM仅在存在“计算机科学”或“古代文明”等概念时才会触发。“当被触发时，特洛伊木马会越狱该模型，使其回答原本会拒绝的有害问题。我们的结果进一步引发了人们对机器学习模型木马攻击的实用性和潜在后果的担忧。



## **34. An Automated, Scalable Machine Learning Model Inversion Assessment Pipeline**

自动化、可扩展的机器学习模型反演评估管道 cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04214v1) [paper-pdf](http://arxiv.org/pdf/2509.04214v1)

**Authors**: Tyler Shumaker, Jessica Carpenter, David Saranchak, Nathaniel D. Bastian

**Abstract**: Machine learning (ML) models have the potential to transform military battlefields, presenting a large external pressure to rapidly incorporate them into operational settings. However, it is well-established that these ML models are vulnerable to a number of adversarial attacks throughout the model deployment pipeline that threaten to negate battlefield advantage. One broad category is privacy attacks (such as model inversion) where an adversary can reverse engineer information from the model, such as the sensitive data used in its training. The ability to quantify the risk of model inversion attacks (MIAs) is not well studied, and there is a lack of automated developmental test and evaluation (DT&E) tools and metrics to quantify the effectiveness of privacy loss of the MIA. The current DT&E process is difficult because ML model inversions can be hard for a human to interpret, subjective when they are interpretable, and difficult to quantify in terms of inversion quality. Additionally, scaling the DT&E process is challenging due to many ML model architectures and data modalities that need to be assessed. In this work, we present a novel DT&E tool that quantifies the risk of data privacy loss from MIAs and introduces four adversarial risk dimensions to quantify privacy loss. Our DT&E pipeline combines inversion with vision language models (VLMs) to improve effectiveness while enabling scalable analysis. We demonstrate effectiveness using multiple MIA techniques and VLMs configured for zero-shot classification and image captioning. We benchmark the pipeline using several state-of-the-art MIAs in the computer vision domain with an image classification task that is typical in military applications. In general, our innovative pipeline extends the current model inversion DT&E capabilities by improving the effectiveness and scalability of the privacy loss analysis in an automated fashion.

摘要: 机器学习（ML）模型有潜力改变军事战场，这给快速将其纳入作战环境带来了巨大的外部压力。然而，众所周知，这些ML模型在整个模型部署管道中容易受到许多对抗攻击，这些攻击可能会抵消战场优势。其中一个广泛的类别是隐私攻击（例如模型倒置），其中对手可以从模型中反向工程信息，例如训练中使用的敏感数据。量化模型倒置攻击（MIA）风险的能力尚未得到充分研究，并且缺乏自动化开发测试和评估（DT & E）工具和指标来量化MIA隐私损失的有效性。当前的DT & E过程很困难，因为ML模型倒置对人类来说可能很难解释，当它们可解释时是主观的，并且很难在倒置质量方面量化。此外，由于需要评估许多ML模型架构和数据模式，扩展DT & E流程具有挑战性。在这项工作中，我们提出了一种新颖的DT & E工具，该工具量化了MIA造成的数据隐私损失的风险，并引入了四个对抗风险维度来量化隐私损失。我们的DT & E管道将倒置与视觉语言模型（VLM）相结合，以提高有效性，同时实现可扩展分析。我们使用多种MIA技术和配置用于零镜头分类和图像字幕的VLM来证明有效性。我们使用计算机视觉领域的几种最先进的MIA对管道进行基准测试，并执行军事应用中典型的图像分类任务。总的来说，我们的创新管道通过以自动化方式提高隐私损失分析的有效性和可扩展性来扩展当前的模型倒置DT & E功能。



## **35. KubeGuard: LLM-Assisted Kubernetes Hardening via Configuration Files and Runtime Logs Analysis**

KubeGuard：通过配置文件和数据库分析的LLM辅助Kubernetes硬化 cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04191v1) [paper-pdf](http://arxiv.org/pdf/2509.04191v1)

**Authors**: Omri Sgan Cohen, Ehud Malul, Yair Meidan, Dudu Mimran, Yuval Elovici, Asaf Shabtai

**Abstract**: The widespread adoption of Kubernetes (K8s) for orchestrating cloud-native applications has introduced significant security challenges, such as misconfigured resources and overly permissive configurations. Failing to address these issues can result in unauthorized access, privilege escalation, and lateral movement within clusters. Most existing K8s security solutions focus on detecting misconfigurations, typically through static analysis or anomaly detection. In contrast, this paper presents KubeGuard, a novel runtime log-driven recommender framework aimed at mitigating risks by addressing overly permissive configurations. KubeGuard is designed to harden K8s environments through two complementary tasks: Resource Creation and Resource Refinement. It leverages large language models (LLMs) to analyze manifests and runtime logs reflecting actual system behavior, using modular prompt-chaining workflows. This approach enables KubeGuard to create least-privilege configurations for new resources and refine existing manifests to reduce the attack surface. KubeGuard's output manifests are presented as recommendations that users (e.g., developers and operators) can review and adopt to enhance cluster security. Our evaluation demonstrates that KubeGuard effectively generates and refines K8s manifests for Roles, NetworkPolicies, and Deployments, leveraging both proprietary and open-source LLMs. The high precision, recall, and F1-scores affirm KubeGuard's practicality as a framework that translates runtime observability into actionable, least-privilege configuration guidance.

摘要: Kubernetes（K8 s）广泛采用来编排云原生应用程序，带来了重大的安全挑战，例如资源配置错误和配置过于宽松。未能解决这些问题可能会导致未经授权的访问、特权升级和集群内的横向移动。大多数现有的K8安全解决方案专注于检测错误配置，通常通过静态分析或异常检测。相比之下，本文介绍了KubeGuard，这是一个新颖的运行时日志驱动的推荐框架，旨在通过解决过度许可的配置来降低风险。KubeGuard旨在通过两项补充任务来强化K8环境：资源创建和资源细化。它利用大型语言模型（LLM）来使用模块化预算链接工作流程来分析反映实际系统行为的清单和运行时日志。这种方法使KubeGuard能够为新资源创建最低特权配置，并改进现有清单以减少攻击面。KubeGuard的输出清单作为用户的推荐（例如，开发者和运营商）可以审查和采用以增强集群安全性。我们的评估表明，KubeGuard可以利用专有和开源LLM有效地生成和改进角色、网络策略和部署的K8清单。高精度、召回率和F1分数证实了KubeGuard作为一个框架的实用性，该框架将运行时可观察性转化为可操作的、最低特权配置指南。



## **36. Privacy Risks in Time Series Forecasting: User- and Record-Level Membership Inference**

时间序列预测中的隐私风险：用户和记录级会员推断 cs.LG

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04169v1) [paper-pdf](http://arxiv.org/pdf/2509.04169v1)

**Authors**: Nicolas Johansson, Tobias Olsson, Daniel Nilsson, Johan Östman, Fazeleh Hoseini

**Abstract**: Membership inference attacks (MIAs) aim to determine whether specific data were used to train a model. While extensively studied on classification models, their impact on time series forecasting remains largely unexplored. We address this gap by introducing two new attacks: (i) an adaptation of multivariate LiRA, a state-of-the-art MIA originally developed for classification models, to the time-series forecasting setting, and (ii) a novel end-to-end learning approach called Deep Time Series (DTS) attack. We benchmark these methods against adapted versions of other leading attacks from the classification setting.   We evaluate all attacks in realistic settings on the TUH-EEG and ELD datasets, targeting two strong forecasting architectures, LSTM and the state-of-the-art N-HiTS, under both record- and user-level threat models. Our results show that forecasting models are vulnerable, with user-level attacks often achieving perfect detection. The proposed methods achieve the strongest performance in several settings, establishing new baselines for privacy risk assessment in time series forecasting. Furthermore, vulnerability increases with longer prediction horizons and smaller training populations, echoing trends observed in large language models.

摘要: 隶属度推理攻击（MIA）旨在确定是否使用特定数据来训练模型。虽然对分类模型进行了广泛研究，但它们对时间序列预测的影响在很大程度上仍未被探索。我们通过引入两种新的攻击来解决这一差距：（i）多元LiRA（最初为分类模型开发的最先进的MIA）适应时间序列预测设置，以及（ii）一种新型的端到端学习方法，称为深度时间序列（RST）攻击。我们将这些方法与分类设置中其他主要攻击的改编版本进行基准测试。   我们在TUH-EEG和ELD数据集上评估现实环境中的所有攻击，针对两种强大的预测架构LSTM和最先进的N-HiTS，在记录级和用户级威胁模型下。我们的结果表明，预测模型很容易受到攻击，用户级攻击通常可以实现完美检测。提出的方法在多种环境下实现了最强的性能，为时间序列预测中的隐私风险评估建立了新的基线。此外，预测视野越长和训练群体越少，脆弱性就会越大，这与大型语言模型中观察到的趋势相呼应。



## **37. Adversarial Bug Reports as a Security Risk in Language Model-Based Automated Program Repair**

对抗性错误报告为基于语言模型的自动程序修复中的安全风险 cs.SE

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.05372v1) [paper-pdf](http://arxiv.org/pdf/2509.05372v1)

**Authors**: Piotr Przymus, Andreas Happe, Jürgen Cito

**Abstract**: Large Language Model (LLM) - based Automated Program Repair (APR) systems are increasingly integrated into modern software development workflows, offering automated patches in response to natural language bug reports. However, this reliance on untrusted user input introduces a novel and underexplored attack surface. In this paper, we investigate the security risks posed by adversarial bug reports -- realistic-looking issue submissions crafted to mislead APR systems into producing insecure or harmful code changes. We develop a comprehensive threat model and conduct an empirical study to evaluate the vulnerability of state-of-the-art APR systems to such attacks. Our demonstration comprises 51 adversarial bug reports generated across a spectrum of strategies, from manual curation to fully automated pipelines. We test these against leading APR model and assess both pre-repair defenses (e.g., LlamaGuard variants, PromptGuard variants, Granite-Guardian, and custom LLM filters) and post-repair detectors (GitHub Copilot, CodeQL). Our findings show that current defenses are insufficient: 90\% of crafted bug reports triggered attacker-aligned patches. The best pre-repair filter blocked only 47\%, while post-repair analysis-often requiring human oversight-was effective in just 58\% of cases. To support scalable security testing, we introduce a prototype framework for automating the generation of adversarial bug reports. Our analysis exposes a structural asymmetry: generating adversarial inputs is inexpensive, while detecting or mitigating them remains costly and error-prone. We conclude with practical recommendations for improving the robustness of APR systems against adversarial misuse and highlight directions for future work on trustworthy automated repair.

摘要: 基于大型语言模型（LLM）的自动程序修复（APL）系统越来越多地集成到现代软件开发工作流程中，提供自动补丁以响应自然语言错误报告。然而，这种对不受信任的用户输入的依赖引入了一种新颖且未充分探索的攻击表面。在本文中，我们调查了对抗性错误报告带来的安全风险--这些报告是为了误导APL系统产生不安全或有害的代码更改而设计的看似真实的问题提交。我们开发了一个全面的威胁模型并进行实证研究来评估最先进的APL系统对此类攻击的脆弱性。我们的演示包括跨越一系列策略（从手动策展到全自动管道）生成的51个对抗性错误报告。我们针对领先的APL模型测试这些并评估修复前防御（例如，LlamaGuard变体、InspectGuard变体、Granite-Guardian和自定义LLM过滤器）和修复后检测器（GitHub Copilot、CodeQL）。我们的调查结果表明，当前的防御措施不足：90%精心设计的错误报告触发了与攻击者对齐的补丁。最好的修复前过滤器仅堵塞了47%，而修复后分析（通常需要人为监督）仅对58%的情况有效。为了支持可扩展的安全测试，我们引入了一个用于自动生成对抗性错误报告的原型框架。我们的分析揭示了结构性不对称：产生对抗性输入的成本很低，而检测或减轻它们的成本仍然很高且容易出错。最后，我们提出了提高APL系统针对对抗性滥用的稳健性的实用建议，并强调了未来值得信赖的自动化修复工作的方向。



## **38. Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks**

先发制人：预先合成类似越狱的指令，以增强LLM安全防范潜在攻击 cs.CL

EMNLP 2025 findings

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2508.20038v3) [paper-pdf](http://arxiv.org/pdf/2508.20038v3)

**Authors**: Sheng Liu, Qiang Sheng, Danding Wang, Yang Li, Guang Yang, Juan Cao

**Abstract**: Despite advances in improving large language model (LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.

摘要: 尽管在改进大型语言模型（LLM）以拒绝回答恶意指令方面取得了进展，但广泛使用的LLM仍然容易受到越狱攻击，攻击者生成的指令分布与安全对齐库不同。新的攻击暴露了LLM无法识别不可见的恶意指令，凸显了训练数据和现实世界攻击之间严重的分布不匹配，迫使开发人员进入反应性补丁周期。为了应对这一挑战，我们提出了IMAGINE，这是一个综合框架，利用嵌入空间分布分析来生成类似越狱的指令。这种方法有效地填补了真实越狱模式和安全调整数据库之间的分布差距。IMAGINE遵循迭代优化过程，在迭代中动态演变文本生成分布，从而通过合成数据示例扩大安全对齐数据分布的覆盖范围。基于通过IMAGINE增强的安全对齐的数据库，我们的框架证明了Qwen 2.5、Llama 3.1和Llama 3.2的攻击成功率显着下降，而不会影响其实用性。



## **39. NeuroBreak: Unveil Internal Jailbreak Mechanisms in Large Language Models**

NeuroBreak：揭开大型语言模型中的内部越狱机制 cs.CR

12 pages, 9 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.03985v1) [paper-pdf](http://arxiv.org/pdf/2509.03985v1)

**Authors**: Chuhan Zhang, Ye Zhang, Bowen Shi, Yuyou Gan, Tianyu Du, Shouling Ji, Dazhan Deng, Yingcai Wu

**Abstract**: In deployment and application, large language models (LLMs) typically undergo safety alignment to prevent illegal and unethical outputs. However, the continuous advancement of jailbreak attack techniques, designed to bypass safety mechanisms with adversarial prompts, has placed increasing pressure on the security defenses of LLMs. Strengthening resistance to jailbreak attacks requires an in-depth understanding of the security mechanisms and vulnerabilities of LLMs. However, the vast number of parameters and complex structure of LLMs make analyzing security weaknesses from an internal perspective a challenging task. This paper presents NeuroBreak, a top-down jailbreak analysis system designed to analyze neuron-level safety mechanisms and mitigate vulnerabilities. We carefully design system requirements through collaboration with three experts in the field of AI security. The system provides a comprehensive analysis of various jailbreak attack methods. By incorporating layer-wise representation probing analysis, NeuroBreak offers a novel perspective on the model's decision-making process throughout its generation steps. Furthermore, the system supports the analysis of critical neurons from both semantic and functional perspectives, facilitating a deeper exploration of security mechanisms. We conduct quantitative evaluations and case studies to verify the effectiveness of our system, offering mechanistic insights for developing next-generation defense strategies against evolving jailbreak attacks.

摘要: 在部署和应用中，大型语言模型（LLM）通常会进行安全调整，以防止非法和不道德的输出。然而，越狱攻击技术的不断进步（旨在通过对抗提示绕过安全机制）给LLM的安全防御带来了越来越大的压力。加强对越狱攻击的抵抗需要深入了解LLM的安全机制和漏洞。然而，LLM的大量参数和复杂结构使得从内部角度分析安全弱点成为一项具有挑战性的任务。本文介绍了NeuroBreak，这是一个自上而下的越狱分析系统，旨在分析神经元级安全机制并缓解漏洞。我们通过与人工智能安全领域的三位专家合作，精心设计系统需求。该系统对各种越狱攻击方法进行了全面分析。通过结合分层表示探测分析，NeuroBreak为模型整个生成步骤的决策过程提供了新颖的视角。此外，该系统支持从语义和功能角度分析关键神经元，促进对安全机制的更深入探索。我们进行定量评估和案例研究来验证我们系统的有效性，为开发针对不断变化的越狱攻击的下一代防御策略提供机械见解。



## **40. Defending LVLMs Against Vision Attacks through Partial-Perception Supervision**

通过部分感知监督保护LVLM免受视觉攻击 cs.CV

Accepted to ICML 2025

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2412.12722v2) [paper-pdf](http://arxiv.org/pdf/2412.12722v2)

**Authors**: Qi Zhou, Tianlin Li, Qing Guo, Dongxia Wang, Yun Lin, Yang Liu, Jin Song Dong

**Abstract**: Recent studies have raised significant concerns regarding the vulnerability of Large Vision Language Models (LVLMs) to maliciously injected or perturbed input images, which can mislead their responses. Existing defense methods show that such vision attacks are sensitive to image modifications especially cropping, using majority voting across responses of modified images as corrected responses. However, these modifications often result in partial images and distort the semantics, which reduces response quality on clean images after voting. Instead of directly using responses from partial images for voting, we investigate using them to supervise the LVLM's responses to the original images. We propose a black-box, training-free method called DPS (Defense through Partial-Perception Supervision). In this approach, the model is prompted using the responses generated by a model that perceives only a partial image. With DPS, the model can adjust its response based on partial image understanding when under attack, while confidently maintaining its original response for clean input. Our findings show that the weak model can supervise the strong model: when faced with an attacked input, the strong model becomes less confident and adjusts its response based on the weak model's partial understanding, effectively defending against the attack. With clean input, it confidently maintains its original response. Empirical experiments show our method outperforms the baseline, cutting the average attack success rate by 76.3% across six datasets on three popular models.

摘要: 最近的研究对大视觉语言模型（LVLM）对恶意注入或干扰的输入图像的脆弱性提出了严重担忧，这可能会误导他们的反应。现有的防御方法表明，此类视觉攻击对图像修改（尤其是裁剪）敏感，使用修改后图像的响应的多数投票作为纠正的响应。然而，这些修改通常会导致部分图像并扭曲语义，从而降低投票后干净图像的响应质量。我们没有直接使用部分图像的响应进行投票，而是研究使用它们来监督LVLM对原始图像的响应。我们提出了一种称为DPS（通过部分感知监督进行防御）的黑匣子、免训练方法。在这种方法中，使用仅感知部分图像的模型生成的响应来提示模型。通过DPS，模型可以在受到攻击时根据部分图像理解调整其响应，同时自信地保持其原始响应以获得干净的输入。我们的研究结果表明，弱模型可以监督强模型：当面对被攻击的输入时，强模型变得不那么自信，并根据弱模型的部分理解调整其响应，有效地防御攻击。通过干净的输入，它自信地保持其原始反应。经验实验表明，我们的方法优于基线，将三种流行模型的六个数据集中的平均攻击成功率降低了76.3%。



## **41. Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs**

在岩石和困难之间：利用道德推理越狱法学硕士 cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.05367v1) [paper-pdf](http://arxiv.org/pdf/2509.05367v1)

**Authors**: Shei Pern Chua, Thai Zhen Leng, Teh Kai Jun, Xiao Li, Xiaolin Hu

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.

摘要: 大型语言模型（LLM）已经进行了安全调整工作，以减轻有害输出。然而，随着LLM的推理变得更加复杂，他们的智能可能会带来新的安全风险。虽然传统的越狱攻击依赖于单步攻击，但动态适应上下文的多回合越狱策略仍然没有得到充分的研究。在这项工作中，我们引入了TRAL（交互式攻击逻辑的电车问题推理），这是一个利用LLM道德推理来绕过其保障措施的框架。TRAL将对抗目标嵌入以电车问题为模型的道德困境中。TriAL展示了开放和封闭源模型的高越狱成功率。我们的研究结果强调了人工智能安全性的一个根本限制：随着模型获得高级推理能力，它们的对齐性质可能会无意中允许更多隐蔽的安全漏洞被利用。TRAL提出了重新评估安全一致监督策略的迫切需要，因为当前的保障措施可能不足以抵御上下文感知的对抗攻击。



## **42. VulRTex: A Reasoning-Guided Approach to Identify Vulnerabilities from Rich-Text Issue Report**

VulRTex：一种从富文本问题报告中识别漏洞的推理引导方法 cs.SE

25 pages, 7 figures, submitting to TOSEM journal

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.03875v1) [paper-pdf](http://arxiv.org/pdf/2509.03875v1)

**Authors**: Ziyou Jiang, Mingyang Li, Guowei Yang, Lin Shi, Qing Wang

**Abstract**: Software vulnerabilities exist in open-source software (OSS), and the developers who discover these vulnerabilities may submit issue reports (IRs) to describe their details. Security practitioners need to spend a lot of time manually identifying vulnerability-related IRs from the community, and the time gap may be exploited by attackers to harm the system. Previously, researchers have proposed automatic approaches to facilitate identifying these vulnerability-related IRs, but these works focus on textual descriptions but lack the comprehensive analysis of IR's rich-text information. In this paper, we propose VulRTex, a reasoning-guided approach to identify vulnerability-related IRs with their rich-text information. In particular, VulRTex first utilizes the reasoning ability of the Large Language Model (LLM) to prepare the Vulnerability Reasoning Database with historical IRs. Then, it retrieves the relevant cases from the prepared reasoning database to generate reasoning guidance, which guides LLM to identify vulnerabilities by reasoning analysis on target IRs' rich-text information. To evaluate the performance of VulRTex, we conduct experiments on 973,572 IRs, and the results show that VulRTex achieves the highest performance in identifying the vulnerability-related IRs and predicting CWE-IDs when the dataset is imbalanced, outperforming the best baseline with +11.0% F1, +20.2% AUPRC, and +10.5% Macro-F1, and 2x lower time cost than baseline reasoning approaches. Furthermore, VulRTex has been applied to identify 30 emerging vulnerabilities across 10 representative OSS projects in 2024's GitHub IRs, and 11 of them are successfully assigned CVE-IDs, which illustrates VulRTex's practicality.

摘要: 软件漏洞存在于开源软件（OSS）中，发现这些漏洞的开发人员可以提交问题报告（IR）来描述其详细信息。安全从业人员需要花费大量时间从社区中手动识别与安全性相关的IR，而攻击者可能会利用这一时间间隔来损害系统。在此之前，研究人员已经提出了自动化的方法来帮助识别这些可识别性相关的IR，但这些作品侧重于文本描述，但缺乏对IR的富文本信息的全面分析。在本文中，我们提出了VulRTex，一种推理引导的方法来识别与其富文本信息相关的可扩展性IR。特别是，VulRTex首先利用大型语言模型（LLM）的推理能力来准备具有历史IR的漏洞推理数据库。然后，从准备好的推理数据库中检索相关案例，生成推理指导，引导LLM通过对目标IR的富文本信息进行推理分析来识别漏洞。为了评估VulRTex的性能，我们对973，572个IR进行了实验，结果表明VulRTex在识别可互换性相关IR和预测数据集不平衡时的CWE-ID方面达到了最高性能，优于最佳基线，分别为+11.0%F1、+20.2%AUPRC和+10. 5% Macro-F1，时间成本比基线推理方法低2倍。此外，VulRTex已被应用于识别2024年GitHub IR中10个代表性OSS项目的30个新兴漏洞，其中11个已成功分配了CVE-ID，这说明了VulRTex的实用性。



## **43. Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models**

使用语言模型生成的对抗性示例进行攻击错误信息检测 cs.CL

Presented at EMNLP 2025

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2410.20940v2) [paper-pdf](http://arxiv.org/pdf/2410.20940v2)

**Authors**: Piotr Przybyła, Euan McGill, Horacio Saggion

**Abstract**: Large language models have many beneficial applications, but can they also be used to attack content-filtering algorithms in social media platforms? We investigate the challenge of generating adversarial examples to test the robustness of text classification algorithms detecting low-credibility content, including propaganda, false claims, rumours and hyperpartisan news. We focus on simulation of content moderation by setting realistic limits on the number of queries an attacker is allowed to attempt. Within our solution (TREPAT), initial rephrasings are generated by large language models with prompts inspired by meaning-preserving NLP tasks, such as text simplification and style transfer. Subsequently, these modifications are decomposed into small changes, applied through beam search procedure, until the victim classifier changes its decision. We perform (1) quantitative evaluation using various prompts, models and query limits, (2) targeted manual assessment of the generated text and (3) qualitative linguistic analysis. The results confirm the superiority of our approach in the constrained scenario, especially in case of long input text (news articles), where exhaustive search is not feasible.

摘要: 大型语言模型有许多有益的应用，但它们也可以用于攻击社交媒体平台中的内容过滤算法吗？我们调查了生成敌对示例的挑战，以测试检测低可信度内容（包括宣传、虚假声明、谣言和超党派新闻）的文本分类算法的稳健性。我们通过对允许攻击者尝试的查询数量设置现实的限制来重点模拟内容审核。在我们的解决方案（TREPAT）中，初始改写由大型语言模型生成，其提示受到保留意义的NLP任务（例如文本简化和风格转移）的启发。随后，这些修改被分解成小的变化，通过束搜索过程应用，直到受害者分类器改变其决定。我们（1）使用各种提示、模型和查询限制执行定量评估，（2）对生成的文本进行有针对性的手动评估，（3）定性语言分析。结果证实了我们的方法在受限场景中的优越性，特别是在长输入文本（新闻文章）的情况下，其中详尽搜索是不可行的。



## **44. PromptCOS: Towards System Prompt Copyright Auditing for LLMs via Content-level Output Similarity**

Observtcos：通过内容级输出相似性实现LLM的系统提示版权审计 cs.CR

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03117v1) [paper-pdf](http://arxiv.org/pdf/2509.03117v1)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Enhao Huang, Shuo Shao, Bingrun Yang, Zhibo Wang, Dacheng Tao, Zhan Qin

**Abstract**: The rapid progress of large language models (LLMs) has greatly enhanced reasoning tasks and facilitated the development of LLM-based applications. A critical factor in improving LLM-based applications is the design of effective system prompts, which significantly impact the behavior and output quality of LLMs. However, system prompts are susceptible to theft and misuse, which could undermine the interests of prompt owners. Existing methods protect prompt copyrights through watermark injection and verification but face challenges due to their reliance on intermediate LLM outputs (e.g., logits), which limits their practical feasibility.   In this paper, we propose PromptCOS, a method for auditing prompt copyright based on content-level output similarity. It embeds watermarks by optimizing the prompt while simultaneously co-optimizing a special verification query and content-level signal marks. This is achieved by leveraging cyclic output signals and injecting auxiliary tokens to ensure reliable auditing in content-only scenarios. Additionally, it incorporates cover tokens to protect the watermark from malicious deletion. For copyright verification, PromptCOS identifies unauthorized usage by comparing the similarity between the suspicious output and the signal mark. Experimental results demonstrate that our method achieves high effectiveness (99.3% average watermark similarity), strong distinctiveness (60.8% greater than the best baseline), high fidelity (accuracy degradation of no more than 0.58%), robustness (resilience against three types of potential attacks), and computational efficiency (up to 98.1% reduction in computational cost). Our code is available at GitHub https://github.com/LianPing-cyber/PromptCOS.

摘要: 大型语言模型（LLM）的快速发展极大地增强了推理任务并促进了基于LLM的应用程序的开发。改进基于LLM的应用程序的一个关键因素是设计有效的系统提示，这将显着影响LLM的行为和输出质量。然而，系统提示很容易被盗窃和滥用，这可能会损害提示所有者的利益。现有方法通过水印注入和验证来保护即时版权，但由于依赖中间LLM输出（例如，logits），这限制了它们的实际可行性。   在本文中，我们提出了Inbox cos，这是一种基于内容级输出相似度的提示版权审计方法。它通过优化提示来嵌入水印，同时协同优化特殊验证查询和内容级信号标记。这是通过利用循环输出信号和注入辅助令牌来实现的，以确保仅内容场景中的可靠审计。此外，它还结合了封面令牌来保护水印免受恶意删除。对于版权验证，Inbox cos通过比较可疑输出和信号标记之间的相似性来识别未经授权的使用。实验结果表明，我们的方法具有高有效性（平均水印相似度为99.3%）、强区分性（比最佳基线高60.8%）、高保真度（准确率下降不超过0.58%）、鲁棒性（对三种潜在攻击的弹性）和计算效率（计算成本降低高达98.1%）。我们的代码可在GitHub https://github.com/LianPing-cyber/PromptCOS上获取。



## **45. EverTracer: Hunting Stolen Large Language Models via Stealthy and Robust Probabilistic Fingerprint**

EverTracer：通过隐秘且稳健的概率指纹追踪被盗的大型语言模型 cs.CR

Accepted by EMNLP2025 Main

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03058v1) [paper-pdf](http://arxiv.org/pdf/2509.03058v1)

**Authors**: Zhenhua Xu, Meng Han, Wenpeng Xing

**Abstract**: The proliferation of large language models (LLMs) has intensified concerns over model theft and license violations, necessitating robust and stealthy ownership verification. Existing fingerprinting methods either require impractical white-box access or introduce detectable statistical anomalies. We propose EverTracer, a novel gray-box fingerprinting framework that ensures stealthy and robust model provenance tracing. EverTracer is the first to repurpose Membership Inference Attacks (MIAs) for defensive use, embedding ownership signals via memorization instead of artificial trigger-output overfitting. It consists of Fingerprint Injection, which fine-tunes the model on any natural language data without detectable artifacts, and Verification, which leverages calibrated probability variation signal to distinguish fingerprinted models. This approach remains robust against adaptive adversaries, including input level modification, and model-level modifications. Extensive experiments across architectures demonstrate EverTracer's state-of-the-art effectiveness, stealthness, and resilience, establishing it as a practical solution for securing LLM intellectual property. Our code and data are publicly available at https://github.com/Xuzhenhua55/EverTracer.

摘要: 大型语言模型（LLM）的激增加剧了人们对模型盗窃和许可证违规的担忧，需要进行强大且隐蔽的所有权验证。现有的指纹识别方法要么需要不切实际的白盒访问，要么引入可检测到的统计异常。我们提出了EverTracer，这是一种新型的灰箱指纹识别框架，可以确保隐蔽且稳健的模型出处追踪。EverTracer是第一个将会员推断攻击（MIA）重新用于防御用途的公司，通过记忆而不是人为的命令输出过度匹配来嵌入所有权信号。它由指纹注入和验证组成，前者在任何自然语言数据上微调模型，而不会检测到伪影，后者利用校准的概率变化信号来区分指纹模型。这种方法对于自适应对手（包括输入级别修改和模型级别修改）仍然具有鲁棒性。跨架构的广泛实验证明了EverTracer最先进的有效性、隐蔽性和弹性，使其成为保护LLM知识产权的实用解决方案。我们的代码和数据可在https://github.com/Xuzhenhua55/EverTracer上公开获取。



## **46. FedAPT: Federated Adversarial Prompt Tuning for Vision-Language Models**

FedAPT：视觉语言模型的联邦对抗提示调优 cs.CV

ACM MM25

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.06992v1) [paper-pdf](http://arxiv.org/pdf/2509.06992v1)

**Authors**: Kun Zhai, Siheng Chen, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Federated Prompt Tuning (FPT) is an efficient method for cross-client collaborative fine-tuning of large Vision-Language Models (VLMs). However, models tuned using FPT are vulnerable to adversarial attacks, leading to misclassification in downstream tasks. In this work, we introduce Federated Adversarial Prompt Tuning (\textbf{FedAPT}), a novel method designed to enhance the adversarial robustness of FPT. We identify a key issue in FedAPT under non-independent and identically distributed (non-IID) settings: a \textit{class information gap} between clients and the global model. Clients rely solely on limited local label information to generate adversarial samples for training, while the global model must defend against adversarial attacks from global labels. To address this issue, we propose a \textbf{class-aware prompt generator} that generates visual prompts from text prompts. This generator is guided by a \emph{Global Label Embedding} (serving as a ``beacon") which encodes cross-client label information to create more globally-aligned visual prompts. Additionally, we propose a \textbf{cross-layer generator sharing} strategy to enhance prompt coupling across different layers of the model, further boosting adversarial robustness. Extensive experiments on multiple image classification datasets demonstrate the superiority of FedAPT in improving adversarial robustness, outperforming existing methods by a large margin. FedAPT also exhibits exceptional generalization in cross-domain and cross-dataset scenarios, indicating its effectiveness in real-world applications.

摘要: 联合提示调优（FPT）是一种用于大型视觉语言模型（VLM）的跨客户端协作微调的有效方法。然而，使用FPT调整的模型很容易受到对抗攻击，导致下游任务中的错误分类。在这项工作中，我们引入了联邦对抗提示调优（\textBF{FedAPT}），这是一种旨在增强FPT对抗鲁棒性的新型方法。我们在非独立和同分布（非IID）设置下发现了FedAPT中的一个关键问题：客户端和全球模型之间的\textit{类信息差距}。客户仅依赖有限的本地标签信息来生成对抗样本进行训练，而全球模型必须防御来自全球标签的对抗攻击。为了解决这个问题，我们提出了一个\textBF{类感知提示生成器}，它从文本提示生成视觉提示。此生成器由\ð {Global Label Embedding}（充当“beacon”）引导，该生成器对跨客户端标签信息进行编码，以创建更加全球一致的视觉提示。此外，我们提出了一种\textBF{跨层生成器共享}策略，以增强模型不同层之间的即时耦合，进一步增强对抗鲁棒性。对多个图像分类数据集的大量实验证明了FedAPT在提高对抗鲁棒性方面的优势，大幅优于现有方法。FedAPT还在跨领域和跨数据集场景中表现出出色的概括性，表明其在现实世界应用程序中的有效性。



## **47. See No Evil: Adversarial Attacks Against Linguistic-Visual Association in Referring Multi-Object Tracking Systems**

看不到邪恶：引用多对象跟踪系统中对语言视觉关联的对抗攻击 cs.CV

12 pages, 1 figure, 3 tables

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.02028v2) [paper-pdf](http://arxiv.org/pdf/2509.02028v2)

**Authors**: Halima Bouzidi, Haoyu Liu, Mohammad Abdullah Al Faruque

**Abstract**: Language-vision understanding has driven the development of advanced perception systems, most notably the emerging paradigm of Referring Multi-Object Tracking (RMOT). By leveraging natural-language queries, RMOT systems can selectively track objects that satisfy a given semantic description, guided through Transformer-based spatial-temporal reasoning modules. End-to-End (E2E) RMOT models further unify feature extraction, temporal memory, and spatial reasoning within a Transformer backbone, enabling long-range spatial-temporal modeling over fused textual-visual representations. Despite these advances, the reliability and robustness of RMOT remain underexplored. In this paper, we examine the security implications of RMOT systems from a design-logic perspective, identifying adversarial vulnerabilities that compromise both the linguistic-visual referring and track-object matching components. Additionally, we uncover a novel vulnerability in advanced RMOT models employing FIFO-based memory, whereby targeted and consistent attacks on their spatial-temporal reasoning introduce errors that persist within the history buffer over multiple subsequent frames. We present VEIL, a novel adversarial framework designed to disrupt the unified referring-matching mechanisms of RMOT models. We show that carefully crafted digital and physical perturbations can corrupt the tracking logic reliability, inducing track ID switches and terminations. We conduct comprehensive evaluations using the Refer-KITTI dataset to validate the effectiveness of VEIL and demonstrate the urgent need for security-aware RMOT designs for critical large-scale applications.

摘要: 图像视觉理解推动了高级感知系统的发展，最引人注目的是参考多对象跟踪（RMOT）的新兴范式。通过利用自然语言查询，RMOT系统可以在基于Transformer的时空推理模块的指导下选择性地跟踪满足给定语义描述的对象。端到端（E2 E）RMOT模型进一步统一了Transformer骨干中的特征提取、时间记忆和空间推理，从而在融合的文本-视觉表示上实现了长距离时空建模。尽管有这些进步，RMOT的可靠性和鲁棒性仍然没有得到充分的研究。在本文中，我们从设计逻辑的角度研究了RMOT系统的安全影响，识别了损害语言视觉引用和跟踪对象匹配组件的对抗漏洞。此外，我们还发现了使用基于FIFO的内存的高级RMOT模型中的一个新漏洞，从而对其时空推理进行有针对性且一致的攻击，从而引入了在历史缓冲区中持续存在的错误。我们提出了VEIL，这是一种新型对抗框架，旨在破坏RMOT模型的统一触发匹配机制。我们表明，精心设计的数字和物理扰动可能会破坏跟踪逻辑的可靠性，导致轨道ID开关和端接。我们使用Refer-KITTI数据集进行全面评估，以验证VEIL的有效性，并证明关键大规模应用对安全感知RMOT设计的迫切需求。



## **48. A Survey: Towards Privacy and Security in Mobile Large Language Models**

调查：移动大型语言模型中的隐私和安全 cs.CR

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02411v1) [paper-pdf](http://arxiv.org/pdf/2509.02411v1)

**Authors**: Honghui Xu, Kaiyang Li, Wei Chen, Danyang Zheng, Zhiyuan Li, Zhipeng Cai

**Abstract**: Mobile Large Language Models (LLMs) are revolutionizing diverse fields such as healthcare, finance, and education with their ability to perform advanced natural language processing tasks on-the-go. However, the deployment of these models in mobile and edge environments introduces significant challenges related to privacy and security due to their resource-intensive nature and the sensitivity of the data they process. This survey provides a comprehensive overview of privacy and security issues associated with mobile LLMs, systematically categorizing existing solutions such as differential privacy, federated learning, and prompt encryption. Furthermore, we analyze vulnerabilities unique to mobile LLMs, including adversarial attacks, membership inference, and side-channel attacks, offering an in-depth comparison of their effectiveness and limitations. Despite recent advancements, mobile LLMs face unique hurdles in achieving robust security while maintaining efficiency in resource-constrained environments. To bridge this gap, we propose potential applications, discuss open challenges, and suggest future research directions, paving the way for the development of trustworthy, privacy-compliant, and scalable mobile LLM systems.

摘要: 移动大型语言模型（LLM）正在彻底改变医疗保健、金融和教育等各个领域，因为它们能够随时执行高级自然语言处理任务。然而，由于这些模型的资源密集型性质和处理数据的敏感性，在移动和边缘环境中部署这些模型会带来与隐私和安全相关的重大挑战。本调查全面概述了与移动LLM相关的隐私和安全问题，系统地对现有解决方案进行分类，例如差异隐私、联合学习和提示加密。此外，我们还分析了移动LLM特有的漏洞，包括对抗性攻击、成员资格推断和侧通道攻击，并对其有效性和局限性进行了深入比较。尽管最近取得了进步，但移动LLM在实现强大的安全性同时在资源有限的环境中保持效率方面面临着独特的障碍。为了弥合这一差距，我们提出了潜在的应用程序，讨论了开放的挑战，并提出了未来的研究方向，为开发值得信赖、符合隐私和可扩展的移动LLM系统铺平了道路。



## **49. Enhancing Reliability in LLM-Integrated Robotic Systems: A Unified Approach to Security and Safety**

提高LLM集成机器人系统的可靠性：统一的安全保障方法 cs.RO

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02163v1) [paper-pdf](http://arxiv.org/pdf/2509.02163v1)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Bräunl, Jin B. Hong

**Abstract**: Integrating large language models (LLMs) into robotic systems has revolutionised embodied artificial intelligence, enabling advanced decision-making and adaptability. However, ensuring reliability, encompassing both security against adversarial attacks and safety in complex environments, remains a critical challenge. To address this, we propose a unified framework that mitigates prompt injection attacks while enforcing operational safety through robust validation mechanisms. Our approach combines prompt assembling, state management, and safety validation, evaluated using both performance and security metrics. Experiments show a 30.8% improvement under injection attacks and up to a 325% improvement in complex environment settings under adversarial conditions compared to baseline scenarios. This work bridges the gap between safety and security in LLM-based robotic systems, offering actionable insights for deploying reliable LLM-integrated mobile robots in real-world settings. The framework is open-sourced with simulation and physical deployment demos at https://llmeyesim.vercel.app/

摘要: 将大型语言模型（LLM）集成到机器人系统中彻底改变了具体人工智能，实现了高级决策和适应性。然而，确保可靠性（包括对抗攻击的安全性和复杂环境中的安全性）仍然是一个严峻的挑战。为了解决这个问题，我们提出了一个统一的框架，该框架可以减轻即时注入攻击，同时通过强大的验证机制强制执行操作安全。我们的方法结合了即时组装、状态管理和安全验证，并使用性能和安全指标进行评估。实验表明，与基线场景相比，在注入攻击下的性能提高了30.8%，在对抗条件下的复杂环境设置中的性能提高了325%。这项工作弥合了基于LLM的机器人系统的安全性和安全性之间的差距，为在现实环境中部署可靠的LLM集成移动机器人提供了可操作的见解。该框架是开源的，在https://llmeyesim.vercel.app/上有模拟和物理部署演示



## **50. Unraveling LLM Jailbreaks Through Safety Knowledge Neurons**

通过安全知识神经元解开LLM越狱事件 cs.AI

10 pages, 6 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01631v1) [paper-pdf](http://arxiv.org/pdf/2509.01631v1)

**Authors**: Chongwen Zhao, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation, a technique known as "Jailbreak." While some studies have achieved defenses against jailbreak attacks by modifying output distributions or detecting harmful content, the exact rationale still remains elusive. In this work, we present a novel neuron-level interpretability method that focuses on the role of safety-related knowledge neurons. Unlike existing approaches, our method projects the model's internal representation into a more consistent and interpretable vocabulary space. We then show that adjusting the activation of safety-related neurons can effectively control the model's behavior with a mean ASR higher than 97%. Building on this insight, we propose SafeTuning, a fine-tuning strategy that reinforces safety-critical neurons to improve model robustness against jailbreaks. SafeTuning consistently reduces attack success rates across multiple LLMs and outperforms all four baseline defenses. These findings offer a new perspective on understanding and defending against jailbreak attacks.

摘要: 大型语言模型（LLM）在各种应用中越来越受到关注。尽管如此，人们越来越担心，因为一些用户试图利用这些模型进行恶意目的，包括合成受控物质和传播虚假信息，一种被称为“越狱”的技术。“虽然一些研究通过修改输出分发或检测有害内容来实现对越狱攻击的防御，但确切的理由仍然难以捉摸。在这项工作中，我们提出了一种新型的神经元水平解释方法，重点关注与安全相关的知识神经元的作用。与现有方法不同，我们的方法将模型的内部表示投影到更一致和可解释的词汇空间中。然后我们表明，调整安全相关神经元的激活可以有效控制模型的行为，平均ASB高于97%。基于这一见解，我们提出了SafeTuning，这是一种微调策略，可以加强安全关键神经元，以提高模型针对越狱的鲁棒性。SafeTuning持续降低多个LLM的攻击成功率，并优于所有四种基线防御。这些发现为理解和防御越狱攻击提供了新的视角。



