# LLM / MLLM（语言模型） - 安全对齐
**update at 2026-01-25 10:36:50**

按分类器置信度从高到低排序。

## **1. Provable Robustness in Multimodal Large Language Models via Feature Space Smoothing**

通过特征空间平滑实现多模态大语言模型的可证明鲁棒性 cs.LG

Under review

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2601.16200v1) [paper-pdf](https://arxiv.org/pdf/2601.16200v1)

**Confidence**: 0.95

**Authors**: Song Xia, Meiwen Ding, Chenqi Kong, Wenhan Yang, Xudong Jiang

**Abstract**: Multimodal large language models (MLLMs) exhibit strong capabilities across diverse applications, yet remain vulnerable to adversarial perturbations that distort their feature representations and induce erroneous predictions. To address this vulnerability, we propose the Feature-space Smoothing (FS) and theoretically prove that FS offers certified robustness on the feature representations of MLLMs. Specifically, FS transforms any feature encoder into a smoothed variant that is guaranteed to maintain a certified lower bound on the feature cosine similarity between clean and adversarial representations under $\ell_2$-bounded attacks. Moreover, we indicate that the value of this Feature Cosine Similarity Bound (FCSB) derived from FS can be improved by enlarging the defined Gaussian robustness score on the vanilla encoder. Building upon this, we introduce the Purifier and Smoothness Mapper (PSM), a plug-and-play module that improves the Gaussian robustness score of MLLMs and thus enhances their certified robustness under FS, without requiring any retraining on MLLMs. We demonstrate that the FS with PSM not only provides a strong theoretical robustness guarantee but also exhibits superior empirical performance compared to adversarial training. Extensive experiments across diverse MLLMs and downstream tasks indicate the effectiveness of the FS-PSM, reducing the Attack Success Rate (ASR) of various white-box attacks from nearly 90\% to about 1\%.

摘要: 多模态大语言模型（MLLMs）在多种应用中展现出强大能力，但仍易受对抗性扰动影响，这些扰动会扭曲其特征表示并导致错误预测。为解决此脆弱性，我们提出特征空间平滑（FS）方法，并从理论上证明FS能为MLLMs的特征表示提供可证明的鲁棒性保证。具体而言，FS将任意特征编码器转换为平滑变体，确保在ℓ₂有界攻击下，干净特征与对抗特征之间的余弦相似度存在可证明的下界。此外，我们指出通过提升原始编码器的高斯鲁棒性评分，可改进FS导出的特征余弦相似度下界（FCSB）值。基于此，我们引入即插即用模块——净化与平滑映射器（PSM），该模块能提升MLLMs的高斯鲁棒性评分，从而增强其在FS框架下的可证明鲁棒性，且无需对MLLMs进行任何重训练。实验表明，结合PSM的FS不仅能提供坚实的理论鲁棒性保证，其经验性能也优于对抗训练。在多种MLLMs和下游任务上的大量实验验证了FS-PSM的有效性，能将各类白盒攻击的成功率（ASR）从近90%降至约1%。



## **2. Attributing and Exploiting Safety Vectors through Global Optimization in Large Language Models**

通过全局优化在大语言模型中归因与利用安全向量 cs.LG

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2601.15801v1) [paper-pdf](https://arxiv.org/pdf/2601.15801v1)

**Confidence**: 0.95

**Authors**: Fengheng Chu, Jiahao Chen, Yuhong Wang, Jun Wang, Zhihui Fu, Shouling Ji, Songze Li

**Abstract**: While Large Language Models (LLMs) are aligned to mitigate risks, their safety guardrails remain fragile against jailbreak attacks. This reveals limited understanding of components governing safety. Existing methods rely on local, greedy attribution that assumes independent component contributions. However, they overlook the cooperative interactions between different components in LLMs, such as attention heads, which jointly contribute to safety mechanisms. We propose \textbf{G}lobal \textbf{O}ptimization for \textbf{S}afety \textbf{V}ector Extraction (GOSV), a framework that identifies safety-critical attention heads through global optimization over all heads simultaneously. We employ two complementary activation repatching strategies: Harmful Patching and Zero Ablation. These strategies identify two spatially distinct sets of safety vectors with consistently low overlap, termed Malicious Injection Vectors and Safety Suppression Vectors, demonstrating that aligned LLMs maintain separate functional pathways for safety purposes. Through systematic analyses, we find that complete safety breakdown occurs when approximately 30\% of total heads are repatched across all models. Building on these insights, we develop a novel inference-time white-box jailbreak method that exploits the identified safety vectors through activation repatching. Our attack substantially outperforms existing white-box attacks across all test models, providing strong evidence for the effectiveness of the proposed GOSV framework on LLM safety interpretability.

摘要: 尽管大语言模型（LLMs）经过对齐以降低风险，但其安全防护机制在面对越狱攻击时仍显脆弱，这表明我们对控制安全性的组件理解有限。现有方法依赖于局部贪婪归因，假设组件贡献相互独立，却忽视了LLMs中不同组件（如注意力头）之间的协同交互作用，这些组件共同构成了安全机制。我们提出了全局优化安全向量提取（GOSV）框架，通过同时对所有注意力头进行全局优化来识别安全关键注意力头。我们采用两种互补的激活重定向策略：有害重定向和零消融。这些策略识别出空间上截然不同且重叠度极低的两组安全向量，分别称为恶意注入向量和安全抑制向量，表明对齐后的LLMs为安全目的保留了独立的功能通路。通过系统分析，我们发现当所有模型中约30%的注意力头被重定向时，会引发完全的安全崩溃。基于这些发现，我们开发了一种新颖的推理时白盒越狱方法，通过激活重定向利用已识别的安全向量。我们的攻击在所有测试模型上均显著优于现有白盒攻击方法，为GOSV框架在LLM安全可解释性方面的有效性提供了有力证据。



## **3. INFA-Guard: Mitigating Malicious Propagation via Infection-Aware Safeguarding in LLM-Based Multi-Agent Systems**

INFA-Guard：基于感染感知防护的LLM多智能体系统恶意传播缓解机制 cs.MA

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.14667v1) [paper-pdf](https://arxiv.org/pdf/2601.14667v1)

**Confidence**: 0.95

**Authors**: Yijin Zhou, Xiaoya Lu, Dongrui Liu, Junchi Yan, Jing Shao

**Abstract**: The rapid advancement of Large Language Model (LLM)-based Multi-Agent Systems (MAS) has introduced significant security vulnerabilities, where malicious influence can propagate virally through inter-agent communication. Conventional safeguards often rely on a binary paradigm that strictly distinguishes between benign and attack agents, failing to account for infected agents i.e., benign entities converted by attack agents. In this paper, we propose Infection-Aware Guard, INFA-Guard, a novel defense framework that explicitly identifies and addresses infected agents as a distinct threat category. By leveraging infection-aware detection and topological constraints, INFA-Guard accurately localizes attack sources and infected ranges. During remediation, INFA-Guard replaces attackers and rehabilitates infected ones, avoiding malicious propagation while preserving topological integrity. Extensive experiments demonstrate that INFA-Guard achieves state-of-the-art performance, reducing the Attack Success Rate (ASR) by an average of 33%, while exhibiting cross-model robustness, superior topological generalization, and high cost-effectiveness.

摘要: 基于大语言模型（LLM）的多智能体系统（MAS）的快速发展带来了显著的安全漏洞，恶意影响可通过智能体间通信病毒式传播。传统防护机制通常依赖严格区分良性智能体与攻击智能体的二元范式，未能考虑被感染智能体（即被攻击智能体转化的良性实体）。本文提出感染感知防护框架INFA-Guard，该创新防御框架将受感染智能体明确定位为独立威胁类别进行处理。通过利用感染感知检测与拓扑约束，INFA-Guard能精准定位攻击源与感染范围。在修复阶段，INFA-Guard替换攻击者并修复受感染智能体，在维护拓扑完整性的同时阻断恶意传播。大量实验表明，INFA-Guard实现了最先进的性能表现，平均降低33%的攻击成功率（ASR），同时展现出跨模型鲁棒性、优越的拓扑泛化能力及高成本效益。



## **4. NeuroFilter: Privacy Guardrails for Conversational LLM Agents**

NeuroFilter：对话式LLM代理的隐私防护栏 cs.CR

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.14660v1) [paper-pdf](https://arxiv.org/pdf/2601.14660v1)

**Confidence**: 0.95

**Authors**: Saswat Das, Ferdinando Fioretto

**Abstract**: This work addresses the computational challenge of enforcing privacy for agentic Large Language Models (LLMs), where privacy is governed by the contextual integrity framework. Indeed, existing defenses rely on LLM-mediated checking stages that add substantial latency and cost, and that can be undermined in multi-turn interactions through manipulation or benign-looking conversational scaffolding. Contrasting this background, this paper makes a key observation: internal representations associated with privacy-violating intent can be separated from benign requests using linear structure. Using this insight, the paper proposes NeuroFilter, a guardrail framework that operationalizes contextual integrity by mapping norm violations to simple directions in the model's activation space, enabling detection even when semantic filters are bypassed. The proposed filter is also extended to capture threats arising during long conversations using the concept of activation velocity, which measures cumulative drift in internal representations across turns. A comprehensive evaluation across over 150,000 interactions and covering models from 7B to 70B parameters, illustrates the strong performance of NeuroFilter in detecting privacy attacks while maintaining zero false positives on benign prompts, all while reducing the computational inference cost by several orders of magnitude when compared to LLM-based agentic privacy defenses.

摘要: 本研究解决了为基于情境完整性的智能体大语言模型（LLM）实施隐私保护的计算挑战。现有防御方案依赖LLM中介的检查阶段，这会带来显著的延迟和成本，并且在多轮交互中可能通过操纵或看似良性的对话框架被破坏。与此背景相对，本文提出关键观察：与侵犯隐私意图相关的内部表征可通过线性结构与良性请求分离。基于这一洞见，本文提出NeuroFilter防护栏框架，通过将规范违反映射到模型激活空间的简单方向来实现情境完整性操作化，即使在语义过滤器被绕过时也能实现检测。该过滤器还通过激活速度概念扩展到捕获长对话中出现的威胁，该概念测量跨轮次内部表征的累积漂移。通过对超过150,000次交互的全面评估，涵盖7B至70B参数的模型，结果表明NeuroFilter在检测隐私攻击方面表现优异，同时在良性提示上保持零误报，与基于LLM的智能体隐私防御相比，计算推理成本降低了数个数量级。



## **5. The Side Effects of Being Smart: Safety Risks in MLLMs' Multi-Image Reasoning**

智能的副作用：MLLMs多图像推理中的安全风险 cs.CV

*15 pages, 5 figures. Introduces MIR-SafetyBench (2,676 instances; 9 multi-image relations). Equal contribution; †Corresponding author. Code/data: https://github.com/thu-coai/MIR-SafetyBench

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.14127v1) [paper-pdf](https://arxiv.org/pdf/2601.14127v1)

**Confidence**: 0.95

**Authors**: Renmiao Chen, Yida Lu, Shiyao Cui, Xuan Ouyang, Victor Shea-Jay Huang, Shumin Zhang, Chengwei Pan, Han Qiu, Minlie Huang

**Abstract**: As Multimodal Large Language Models (MLLMs) acquire stronger reasoning capabilities to handle complex, multi-image instructions, this advancement may pose new safety risks. We study this problem by introducing MIR-SafetyBench, the first benchmark focused on multi-image reasoning safety, which consists of 2,676 instances across a taxonomy of 9 multi-image relations. Our extensive evaluations on 19 MLLMs reveal a troubling trend: models with more advanced multi-image reasoning can be more vulnerable on MIR-SafetyBench. Beyond attack success rates, we find that many responses labeled as safe are superficial, often driven by misunderstanding or evasive, non-committal replies. We further observe that unsafe generations exhibit lower attention entropy than safe ones on average. This internal signature suggests a possible risk that models may over-focus on task solving while neglecting safety constraints. Our code and data are available at https://github.com/thu-coai/MIR-SafetyBench.

摘要: 随着多模态大语言模型（MLLMs）获得更强的推理能力以处理复杂的多图像指令，这一进步可能带来新的安全风险。我们通过引入首个专注于多图像推理安全的基准测试MIR-SafetyBench来研究此问题，该基准包含2,676个实例，涵盖9种多图像关系分类。我们对19个MLLMs的广泛评估揭示了一个令人担忧的趋势：具有更先进多图像推理能力的模型在MIR-SafetyBench上可能更加脆弱。除了攻击成功率外，我们发现许多被标记为安全的回答是表面化的，通常源于误解或回避性的非承诺回复。我们进一步观察到，不安全的生成平均比安全的生成具有更低的注意力熵。这一内部特征表明，模型可能过度专注于任务解决而忽视安全约束。我们的代码和数据可在https://github.com/thu-coai/MIR-SafetyBench获取。



## **6. Activation-Space Anchored Access Control for Multi-Class Permission Reasoning in Large Language Models**

基于激活空间锚定的多类别权限推理访问控制框架 cs.CL

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.13630v1) [paper-pdf](https://arxiv.org/pdf/2601.13630v1)

**Confidence**: 0.95

**Authors**: Zhaopeng Zhang, Pengcheng Sun, Lan Zhang, Chen Tang, Jiewei Lai, Yunhao Wang, Hui Jin

**Abstract**: Large language models (LLMs) are increasingly deployed over knowledge bases for efficient knowledge retrieval and question answering. However, LLMs can inadvertently answer beyond a user's permission scope, leaking sensitive content, thus making it difficult to deploy knowledge-base QA under fine-grained access control requirements. In this work, we identify a geometric regularity in intermediate activations: for the same query, representations induced by different permission scopes cluster distinctly and are readily separable. Building on this separability, we propose Activation-space Anchored Access Control (AAAC), a training-free framework for multi-class permission control. AAAC constructs an anchor bank, with one permission anchor per class, from a small offline sample set and requires no fine-tuning. At inference time, a multi-anchor steering mechanism redirects each query's activations toward the anchor-defined authorized region associated with the current user, thereby suppressing over-privileged generations by design. Finally, extensive experiments across three LLM families demonstrate that AAAC reduces permission violation rates by up to 86.5% and prompt-based attack success rates by 90.7%, while improving response usability with minor inference overhead compared to baselines.

摘要: 大型语言模型（LLMs）正越来越多地部署在知识库上，以实现高效的知识检索和问答。然而，LLMs可能无意中回答超出用户权限范围的问题，泄露敏感内容，这使得在细粒度访问控制要求下部署知识库问答系统变得困难。本研究发现了中间激活的几何规律性：对于相同查询，不同权限范围诱导的表征会形成明显分离的聚类。基于这种可分离性，我们提出了激活空间锚定访问控制（AAAC），一种无需训练的多类别权限控制框架。AAAC通过少量离线样本集构建权限锚点库（每个类别对应一个权限锚点），无需微调。在推理时，多锚点引导机制将每个查询的激活重定向至与当前用户关联的锚点定义授权区域，从而从设计上抑制越权生成。最后，在三个LLM系列上的大量实验表明，与基线方法相比，AAAC将权限违规率降低高达86.5%，基于提示的攻击成功率降低90.7%，同时以较小的推理开销提升了响应可用性。



## **7. Prompt Injection Mitigation with Agentic AI, Nested Learning, and AI Sustainability via Semantic Caching**

基于智能体AI、嵌套学习和语义缓存的提示注入缓解与AI可持续性 cs.AI

33 pages, 19 figures

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.13186v1) [paper-pdf](https://arxiv.org/pdf/2601.13186v1)

**Confidence**: 0.95

**Authors**: Diego Gosmar, Deborah A. Dahl

**Abstract**: Prompt injection remains a central obstacle to the safe deployment of large language models, particularly in multi-agent settings where intermediate outputs can propagate or amplify malicious instructions. Building on earlier work that introduced a four-metric Total Injection Vulnerability Score (TIVS), this paper extends the evaluation framework with semantic similarity-based caching and a fifth metric (Observability Score Ratio) to yield TIVS-O, investigating how defence effectiveness interacts with transparency in a HOPE-inspired Nested Learning architecture. The proposed system combines an agentic pipeline with Continuum Memory Systems that implement semantic similarity-based caching across 301 synthetically generated injection-focused prompts drawn from ten attack families, while a fourth agent performs comprehensive security analysis using five key performance indicators. In addition to traditional injection metrics, OSR quantifies the richness and clarity of security-relevant reasoning exposed by each agent, enabling an explicit analysis of trade-offs between strict mitigation and auditability. Experiments show that the system achieves secure responses with zero high-risk breaches, while semantic caching delivers substantial computational savings, achieving a 41.6% reduction in LLM calls and corresponding decreases in latency, energy consumption, and carbon emissions. Five TIVS-O configurations reveal optimal trade-offs between mitigation strictness and forensic transparency. These results indicate that observability-aware evaluation can reveal non-monotonic effects within multi-agent pipelines and that memory-augmented agents can jointly maximize security robustness, real-time performance, operational cost savings, and environmental sustainability without modifying underlying model weights, providing a production-ready pathway for secure and green LLM deployments.

摘要: 提示注入仍然是安全部署大语言模型的核心障碍，特别是在多智能体环境中，中间输出可能传播或放大恶意指令。本文在早期引入四指标总注入漏洞评分（TIVS）的研究基础上，通过基于语义相似性的缓存和第五个指标（可观测性评分比）扩展评估框架，形成TIVS-O，研究在受HOPE启发的嵌套学习架构中防御效果与透明度的相互作用。所提出的系统结合了智能体流水线和连续记忆系统，后者在来自十个攻击家族的301个合成生成的注入提示上实现基于语义相似性的缓存，同时第四个智能体使用五个关键性能指标进行全面的安全分析。除了传统的注入指标外，OSR量化了每个智能体暴露的安全相关推理的丰富性和清晰度，从而能够明确分析严格缓解与可审计性之间的权衡。实验表明，该系统实现了零高风险漏洞的安全响应，同时语义缓存带来了显著的计算节省，将LLM调用减少了41.6%，并相应降低了延迟、能耗和碳排放。五种TIVS-O配置揭示了缓解严格性与取证透明度之间的最佳权衡。这些结果表明，可观测性感知评估可以揭示多智能体流水线内的非单调效应，并且记忆增强的智能体可以共同最大化安全鲁棒性、实时性能、运营成本节约和环境可持续性，而无需修改底层模型权重，为安全和绿色的LLM部署提供了生产就绪的途径。



## **8. Adversarial Alignment: Ensuring Value Consistency in Large Language Models for Sensitive Domains**

对抗性对齐：确保大型语言模型在敏感领域的价值一致性 cs.CL

13 pages, 5 figures

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2601.13137v2) [paper-pdf](https://arxiv.org/pdf/2601.13137v2)

**Confidence**: 0.95

**Authors**: Yuan Gao, Zhigang Liu, Xinyu Yao, Bo Chen, Xiaobing Zhao

**Abstract**: With the wide application of large language models (LLMs), the problems of bias and value inconsistency in sensitive domains have gradually emerged, especially in terms of race, society and politics. In this paper, we propose an adversarial alignment framework, which enhances the value consistency of the model in sensitive domains through continued pre-training, instruction fine-tuning and adversarial training. In adversarial training, we use the Attacker to generate controversial queries, the Actor to generate responses with value consistency, and the Critic to filter and ensure response quality. Furthermore, we train a Value-Consistent Large Language Model, VC-LLM, for sensitive domains, and construct a bilingual evaluation dataset in Chinese and English. The experimental results show that VC-LLM performs better than the existing mainstream models in both Chinese and English tests, verifying the effectiveness of the method. Warning: This paper contains examples of LLMs that are offensive or harmful in nature.

摘要: 随着大型语言模型（LLMs）的广泛应用，其在敏感领域中的偏见与价值不一致问题逐渐显现，尤其在种族、社会和政治方面。本文提出一种对抗性对齐框架，通过持续预训练、指令微调和对抗训练，增强模型在敏感领域的价值一致性。在对抗训练中，我们使用攻击者（Attacker）生成争议性查询，执行者（Actor）生成具有价值一致性的回应，评判者（Critic）则负责筛选并确保回应质量。此外，我们针对敏感领域训练了一个价值一致性大型语言模型VC-LLM，并构建了中英双语评估数据集。实验结果表明，VC-LLM在中英文测试中均优于现有主流模型，验证了该方法的有效性。警告：本文包含具有冒犯性或有害性质的大型语言模型示例。



## **9. Taming Various Privilege Escalation in LLM-Based Agent Systems: A Mandatory Access Control Framework**

驯服基于LLM的智能体系统中的各类权限提升：一种强制访问控制框架 cs.CR

**SubmitDate**: 2026-01-17    [abs](http://arxiv.org/abs/2601.11893v1) [paper-pdf](https://arxiv.org/pdf/2601.11893v1)

**Confidence**: 0.95

**Authors**: Zimo Ji, Daoyuan Wu, Wenyuan Jiang, Pingchuan Ma, Zongjie Li, Yudong Gao, Shuai Wang, Yingjiu Li

**Abstract**: Large Language Model (LLM)-based agent systems are increasingly deployed for complex real-world tasks but remain vulnerable to natural language-based attacks that exploit over-privileged tool use. This paper aims to understand and mitigate such attacks through the lens of privilege escalation, defined as agent actions exceeding the least privilege required for a user's intended task. Based on a formal model of LLM agent systems, we identify novel privilege escalation scenarios, particularly in multi-agent systems, including a variant akin to the classic confused deputy problem. To defend against both known and newly demonstrated privilege escalation, we propose SEAgent, a mandatory access control (MAC) framework built upon attribute-based access control (ABAC). SEAgent monitors agent-tool interactions via an information flow graph and enforces customizable security policies based on entity attributes. Our evaluations show that SEAgent effectively blocks various privilege escalation while maintaining a low false positive rate and negligible system overhead. This demonstrates its robustness and adaptability in securing LLM-based agent systems.

摘要: 基于大语言模型（LLM）的智能体系统正日益广泛地应用于复杂的现实世界任务，但仍易受利用工具过度授权而发起的自然语言攻击。本文旨在通过权限提升的视角来理解和缓解此类攻击，权限提升被定义为智能体行为超出了用户预期任务所需的最小权限。基于对LLM智能体系统的形式化建模，我们识别出新颖的权限提升场景，特别是在多智能体系统中，包括一种类似于经典“困惑代理”问题的变体。为防御已知及新发现的权限提升攻击，我们提出了SEAgent，这是一个构建在基于属性的访问控制（ABAC）之上的强制访问控制（MAC）框架。SEAgent通过信息流图监控智能体与工具的交互，并基于实体属性执行可定制的安全策略。我们的评估表明，SEAgent能有效阻断各类权限提升，同时保持较低的误报率和可忽略的系统开销，这证明了其在保护基于LLM的智能体系统方面的鲁棒性和适应性。



## **10. SD-RAG: A Prompt-Injection-Resilient Framework for Selective Disclosure in Retrieval-Augmented Generation**

SD-RAG：检索增强生成中用于选择性披露的抗提示注入攻击框架 cs.CR

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11199v1) [paper-pdf](https://arxiv.org/pdf/2601.11199v1)

**Confidence**: 0.95

**Authors**: Aiman Al Masoud, Marco Arazzi, Antonino Nocera

**Abstract**: Retrieval-Augmented Generation (RAG) has attracted significant attention due to its ability to combine the generative capabilities of Large Language Models (LLMs) with knowledge obtained through efficient retrieval mechanisms over large-scale data collections. Currently, the majority of existing approaches overlook the risks associated with exposing sensitive or access-controlled information directly to the generation model. Only a few approaches propose techniques to instruct the generative model to refrain from disclosing sensitive information; however, recent studies have also demonstrated that LLMs remain vulnerable to prompt injection attacks that can override intended behavioral constraints. For these reasons, we propose a novel approach to Selective Disclosure in Retrieval-Augmented Generation, called SD-RAG, which decouples the enforcement of security and privacy constraints from the generation process itself. Rather than relying on prompt-level safeguards, SD-RAG applies sanitization and disclosure controls during the retrieval phase, prior to augmenting the language model's input. Moreover, we introduce a semantic mechanism to allow the ingestion of human-readable dynamic security and privacy constraints together with an optimized graph-based data model that supports fine-grained, policy-aware retrieval. Our experimental evaluation demonstrates the superiority of SD-RAG over baseline existing approaches, achieving up to a $58\%$ improvement in the privacy score, while also showing a strong resilience to prompt injection attacks targeting the generative model.

摘要: 检索增强生成（RAG）因其能够将大型语言模型（LLMs）的生成能力与通过大规模数据集合的高效检索机制获取的知识相结合而受到广泛关注。目前，大多数现有方法忽视了将敏感或访问控制信息直接暴露给生成模型所带来的风险。仅有少数方法提出了指导生成模型避免披露敏感信息的技术；然而，最近的研究也表明，LLMs仍然容易受到提示注入攻击的影响，这些攻击可能覆盖预期的行为约束。基于这些原因，我们提出了一种用于检索增强生成中选择性披露的新方法，称为SD-RAG，该方法将安全和隐私约束的执行与生成过程本身解耦。SD-RAG不依赖提示级别的安全措施，而是在检索阶段、在增强语言模型输入之前应用净化和披露控制。此外，我们引入了一种语义机制，允许摄入人类可读的动态安全和隐私约束，并采用优化的基于图的数据模型，支持细粒度的、策略感知的检索。我们的实验评估表明，SD-RAG优于现有的基线方法，在隐私得分上实现了高达58%的提升，同时对针对生成模型的提示注入攻击表现出强大的抗性。



## **11. Can LLM Infer Risk Information From MCP Server System Logs?**

LLM能否从MCP服务器系统日志中推断风险信息？ cs.CR

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2511.05867v3) [paper-pdf](https://arxiv.org/pdf/2511.05867v3)

**Confidence**: 0.95

**Authors**: Jiayi Fu, Yuansen Zhang, Yinggui Wang

**Abstract**: Large Language Models (LLMs) demonstrate strong capabilities in solving complex tasks when integrated with external tools. The Model Context Protocol (MCP) has become a standard interface for enabling such tool-based interactions. However, these interactions introduce substantial security concerns, particularly when the MCP server is compromised or untrustworthy. While prior benchmarks primarily focus on prompt injection attacks or analyze the vulnerabilities of LLM-MCP interaction trajectories, limited attention has been given to the underlying system logs associated with malicious MCP servers. To address this gap, we present the first synthetic benchmark for evaluating LLMs' ability to identify security risks from system logs. We define nine categories of MCP server risks and generate 1,800 synthetic system logs using ten state-of-the-art LLMs. These logs are embedded in the return values of 243 curated MCP servers, yielding a dataset of 2,421 chat histories for training and 471 queries for evaluation. Our pilot experiments reveal that smaller models often fail to detect risky system logs, leading to high false negatives. While models trained with supervised fine-tuning (SFT) tend to over-flag benign logs, resulting in elevated false positives, Reinforcement Learning with Verifiable Reward (RLVR) offers a better precision-recall balance. In particular, after training with Group Relative Policy Optimization (GRPO), Llama3.1-8B-Instruct achieves 83 percent accuracy, surpassing the best-performing large remote model by 9 percentage points. Fine-grained, per-category analysis further underscores the effectiveness of reinforcement learning in enhancing LLM safety within the MCP framework. Code and data are available at https://github.com/PorUna-byte/MCP-RiskCue.

摘要: 大型语言模型（LLMs）在与外部工具集成时展现出解决复杂任务的强大能力。模型上下文协议（MCP）已成为实现此类基于工具交互的标准接口。然而，这些交互带来了重大的安全隐患，尤其是在MCP服务器被入侵或不可信的情况下。虽然现有基准测试主要关注提示注入攻击或分析LLM-MCP交互轨迹的漏洞，但对恶意MCP服务器相关底层系统日志的关注有限。为填补这一空白，我们提出了首个用于评估LLMs从系统日志中识别安全风险能力的合成基准。我们定义了九类MCP服务器风险，并使用十种最先进的LLMs生成了1,800条合成系统日志。这些日志嵌入在243个精选MCP服务器的返回值中，形成了包含2,421条训练用对话历史和471条评估查询的数据集。初步实验表明，较小模型常无法检测风险系统日志，导致高假阴性。虽然通过监督微调（SFT）训练的模型倾向于过度标记良性日志，造成假阳性率升高，但可验证奖励的强化学习（RLVR）提供了更好的精确率-召回率平衡。特别是经过组相对策略优化（GRPO）训练后，Llama3.1-8B-Instruct达到83%的准确率，超越性能最佳的大型远程模型9个百分点。细粒度的分类别分析进一步印证了强化学习在提升MCP框架内LLM安全性方面的有效性。代码和数据可在https://github.com/PorUna-byte/MCP-RiskCue获取。



## **12. From Defender to Devil? Unintended Risk Interactions Induced by LLM Defenses**

从守护者到魔鬼？LLM防御机制引发的意外风险交互 cs.CR

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2510.07968v2) [paper-pdf](https://arxiv.org/pdf/2510.07968v2)

**Confidence**: 0.95

**Authors**: Xiangtao Meng, Tianshuo Cong, Li Wang, Wenyu Chen, Zheng Li, Shanqing Guo, Xiaoyun Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable performance across various applications, but their deployment in real-world settings faces several risks, including jailbreak attacks and privacy leaks. To mitigate these risks, numerous defense strategies have been proposed. However, most existing studies assess these defenses in isolation and ignore their effects on other risk dimensions. In this work, we introduce a new cross-risk evaluation paradigm and take the first step in investigating unintended interactions among defenses in LLMs. Specifically, we focus on the interplay between safety, fairness, and privacy. To this end, we propose CrossRiskEval, a framework that systematically characterizes how a defense designed for one risk (e.g., safety) affects others (e.g., fairness or privacy). We conduct extensive empirical studies and mechanistic analyses on 14 LLMs with deployed defenses, covering 12 defense strategies. Our results show that defenses targeting a single risk often cause measurable effects on other risks. These effects vary in direction and magnitude across a range of factors (e.g., models, tasks, and defense strategies), and are often asymmetric across risk pairs. Furthermore, our mechanistic analysis shows that these interactions are not random: they arise from conflict-entangled neurons, which are shared internal representations that contribute in opposite ways to different risks. Adjusting one risk therefore perturbs these representations and leads to systematic changes in non-target risks. These findings reveal the limits of single-risk evaluation and highlight the need for holistic and interaction-aware assessment when designing and deploying LLM defenses.

摘要: 大型语言模型（LLMs）在各种应用中展现出卓越性能，但其在现实场景中的部署面临多重风险，包括越狱攻击和隐私泄露。为缓解这些风险，研究者提出了众多防御策略。然而，现有研究大多孤立评估这些防御措施，忽视了它们对其他风险维度的影响。本研究提出一种新的跨风险评估范式，首次系统探究LLM防御机制间的意外交互作用，特别关注安全性、公平性与隐私性之间的相互影响。为此，我们开发了CrossRiskEval框架，系统分析针对单一风险（如安全性）的防御措施如何影响其他风险（如公平性或隐私性）。通过对14个部署防御机制的LLM开展大规模实证研究与机理分析，涵盖12种防御策略，我们发现：针对单一风险的防御措施常对其他风险产生可观测的影响；这些影响的方向和强度受模型、任务及防御策略等多重因素调节，且在风险对之间常呈现不对称性。机理分析进一步表明，这些交互并非随机产生，而是源于冲突纠缠神经元——即对不同风险产生相反作用的共享内部表征。调整某一风险会扰动这些表征，进而导致非目标风险的系统性变化。这些发现揭示了单一风险评估的局限性，强调在设计与部署LLM防御机制时，需要采用整体性且关注交互效应的评估方法。



## **13. Panacea: Mitigating Harmful Fine-tuning for Large Language Models via Post-fine-tuning Perturbation**

Panacea：通过后微调扰动缓解大型语言模型的有害微调 cs.CL

Accepted by NeruIPS 2025

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2501.18100v2) [paper-pdf](https://arxiv.org/pdf/2501.18100v2)

**Confidence**: 0.95

**Authors**: Yibo Wang, Tiansheng Huang, Li Shen, Huanjin Yao, Haotian Luo, Rui Liu, Naiqiang Tan, Jiaxing Huang, Dacheng Tao

**Abstract**: Harmful fine-tuning attack introduces significant security risks to the fine-tuning services. Main-stream defenses aim to vaccinate the model such that the later harmful fine-tuning attack is less effective. However, our evaluation results show that such defenses are fragile--with a few fine-tuning steps, the model still can learn the harmful knowledge. To this end, we do further experiment and find that an embarrassingly simple solution--adding purely random perturbations to the fine-tuned model, can recover the model from harmful behaviors, though it leads to a degradation in the model's fine-tuning performance. To address the degradation of fine-tuning performance, we further propose Panacea, which optimizes an adaptive perturbation that will be applied to the model after fine-tuning. Panacea maintains model's safety alignment performance without compromising downstream fine-tuning performance. Comprehensive experiments are conducted on different harmful ratios, fine-tuning tasks and mainstream LLMs, where the average harmful scores are reduced by up-to 21.2%, while maintaining fine-tuning performance. As a by-product, we analyze the adaptive perturbation and show that different layers in various LLMs have distinct safety affinity, which coincide with finding from several previous study. Source code available at https://github.com/w-yibo/Panacea.

摘要: 有害微调攻击对微调服务构成重大安全风险。主流防御方法旨在为模型接种疫苗，使后续的有害微调攻击效果降低。然而，我们的评估结果表明此类防御措施较为脆弱——经过少量微调步骤后，模型仍能学习有害知识。为此，我们进一步实验发现一个极其简单的解决方案——向微调后的模型添加纯随机扰动，可以使模型从有害行为中恢复，尽管这会导致模型微调性能下降。为解决微调性能下降问题，我们进一步提出Panacea方法，该方法优化自适应扰动，将在微调后应用于模型。Panacea在保持模型安全对齐性能的同时，不损害下游微调性能。我们在不同有害比例、微调任务和主流LLM上进行了全面实验，平均有害分数最多降低21.2%，同时保持微调性能。作为副产品，我们分析了自适应扰动，发现不同LLM的各层具有不同的安全亲和性，这与先前多项研究的发现一致。源代码可在https://github.com/w-yibo/Panacea获取。



## **14. Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs**

目标对齐：提取对齐大语言模型的安全分类器 cs.CR

Accepted to 2026 IEEE Secure and Trustworthy Machine Learning Conference (SaTML)

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2501.16534v3) [paper-pdf](https://arxiv.org/pdf/2501.16534v3)

**Confidence**: 0.95

**Authors**: Jean-Charles Noirot Ferrand, Yohan Beugin, Eric Pauley, Ryan Sheatsley, Patrick McDaniel

**Abstract**: Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we introduce and evaluate a new technique for jailbreak attacks. We observe that alignment embeds a safety classifier in the LLM responsible for deciding between refusal and compliance, and seek to extract an approximation of this classifier: a surrogate classifier. To this end, we build candidate classifiers from subsets of the LLM. We first evaluate the degree to which candidate classifiers approximate the LLM's safety classifier in benign and adversarial settings. Then, we attack the candidates and measure how well the resulting adversarial inputs transfer to the LLM. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find that attacks mounted on the surrogate classifiers can be transferred to the LLM with high success. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70% with half the memory footprint and runtime -- a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is an effective and efficient means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks. The code is available at https://github.com/jcnf0/targeting-alignment.

摘要: 大语言模型（LLM）的对齐旨在强制执行安全等准则。然而，当面对通过修改输入诱导不安全输出的越狱攻击时，这种对齐会失效。本文提出并评估了一种新的越狱攻击技术。我们观察到，对齐过程在LLM中嵌入了一个负责在拒绝与遵从之间做出决策的安全分类器，并试图提取该分类器的近似版本：代理分类器。为此，我们从LLM的子集中构建候选分类器。首先，我们在良性及对抗性环境中评估候选分类器对LLM安全分类器的近似程度。随后，我们攻击这些候选分类器，并测量所得对抗性输入向LLM的迁移效果。评估结果表明，最佳候选分类器仅需使用模型架构的20%即可实现高精度一致性（F1分数超过80%）。此外，我们发现针对代理分类器发起的攻击能够以高成功率迁移至原始LLM。例如，仅使用Llama 2模型50%参数的代理分类器实现了70%的攻击成功率（ASR），同时内存占用和运行时间减半——这相较于直接攻击LLM（我们仅观察到22%的ASR）是显著提升。这些结果表明，提取代理分类器是建模（进而应对）对齐模型越狱攻击脆弱性的高效方法。代码发布于https://github.com/jcnf0/targeting-alignment。



## **15. Improving Methodologies for LLM Evaluations Across Global Languages**

改进跨全球语言的LLM评估方法 cs.AI

Author names have been organised by country, and in alphabetical order within countries

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2601.15706v1) [paper-pdf](https://arxiv.org/pdf/2601.15706v1)

**Confidence**: 0.95

**Authors**: Akriti Vij, Benjamin Chua, Darshini Ramiah, En Qi Ng, Mahran Morsidi, Naga Nikshith Gangarapu, Sharmini Johnson, Vanessa Wilfred, Vikneswaran Kumaran, Wan Sie Lee, Wenzhuo Yang, Yongsen Zheng, Bill Black, Boming Xia, Frank Sun, Hao Zhang, Qinghua Lu, Suyu Ma, Yue Liu, Chi-kiu Lo, Fatemeh Azadi, Isar Nejadgholi, Sowmya Vajjala, Agnes Delaborde, Nicolas Rolin, Tom Seimandi, Akiko Murakami, Haruto Ishi, Satoshi Sekine, Takayuki Semitsu, Tasuku Sasaki, Angela Kinuthia, Jean Wangari, Michael Michie, Stephanie Kasaon, Hankyul Baek, Jaewon Noh, Kihyuk Nam, Sang Seo, Sungpil Shin, Taewhi Lee, Yongsu Kim, Daisy Newbold-Harrop, Jessica Wang, Mahmoud Ghanem, Vy Hong

**Abstract**: As frontier AI models are deployed globally, it is essential that their behaviour remains safe and reliable across diverse linguistic and cultural contexts. To examine how current model safeguards hold up in such settings, participants from the International Network for Advanced AI Measurement, Evaluation and Science, including representatives from Singapore, Japan, Australia, Canada, the EU, France, Kenya, South Korea and the UK conducted a joint multilingual evaluation exercise. Led by Singapore AISI, two open-weight models were tested across ten languages spanning high and low resourced groups: Cantonese English, Farsi, French, Japanese, Korean, Kiswahili, Malay, Mandarin Chinese and Telugu. Over 6,000 newly translated prompts were evaluated across five harm categories (privacy, non-violent crime, violent crime, intellectual property and jailbreak robustness), using both LLM-as-a-judge and human annotation.   The exercise shows how safety behaviours can vary across languages. These include differences in safeguard robustness across languages and harm types and variation in evaluator reliability (LLM-as-judge vs. human review). Further, it also generated methodological insights for improving multilingual safety evaluations, such as the need for culturally contextualised translations, stress-tested evaluator prompts and clearer human annotation guidelines. This work represents an initial step toward a shared framework for multilingual safety testing of advanced AI systems and calls for continued collaboration with the wider research community and industry.

摘要: 随着前沿AI模型在全球部署，确保其在多样化的语言和文化环境中保持安全可靠的行为至关重要。为检验当前模型安全措施在此类环境中的表现，来自国际先进AI测量、评估与科学网络的参与者（包括新加坡、日本、澳大利亚、加拿大、欧盟、法国、肯尼亚、韩国和英国的代表）联合开展了一项多语言评估实践。在新加坡AISI的牵头下，对两个开源模型进行了涵盖高资源和低资源群体的十种语言测试：粤语英语、波斯语、法语、日语、韩语、斯瓦希里语、马来语、普通话和泰卢固语。通过LLM-as-a-judge和人工标注两种方式，对超过6,000条新翻译的提示词在五个危害类别（隐私、非暴力犯罪、暴力犯罪、知识产权和越狱鲁棒性）进行了评估。该实践揭示了安全行为在不同语言间的差异，包括安全措施鲁棒性在语言和危害类型上的区别，以及评估者可靠性（LLM-as-judge与人工评审）的差异。此外，研究还提出了改进多语言安全评估的方法学见解，例如需要文化情境化的翻译、经过压力测试的评估提示词以及更清晰的人工标注指南。这项工作代表了建立先进AI系统多语言安全测试共享框架的初步尝试，并呼吁与更广泛的研究界和行业持续合作。



## **16. AgenTRIM: Tool Risk Mitigation for Agentic AI**

AgenTRIM：面向智能体AI的工具风险缓解框架 cs.CR

Under review

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2601.12449v1) [paper-pdf](https://arxiv.org/pdf/2601.12449v1)

**Confidence**: 0.95

**Authors**: Roy Betser, Shamik Bose, Amit Giloni, Chiara Picardi, Sindhu Padakandla, Roman Vainshtein

**Abstract**: AI agents are autonomous systems that combine LLMs with external tools to solve complex tasks. While such tools extend capability, improper tool permissions introduce security risks such as indirect prompt injection and tool misuse. We characterize these failures as unbalanced tool-driven agency. Agents may retain unnecessary permissions (excessive agency) or fail to invoke required tools (insufficient agency), amplifying the attack surface and reducing performance. We introduce AgenTRIM, a framework for detecting and mitigating tool-driven agency risks without altering an agent's internal reasoning. AgenTRIM addresses these risks through complementary offline and online phases. Offline, AgenTRIM reconstructs and verifies the agent's tool interface from code and execution traces. At runtime, it enforces per-step least-privilege tool access through adaptive filtering and status-aware validation of tool calls. Evaluating on the AgentDojo benchmark, AgenTRIM substantially reduces attack success while maintaining high task performance. Additional experiments show robustness to description-based attacks and effective enforcement of explicit safety policies. Together, these results demonstrate that AgenTRIM provides a practical, capability-preserving approach to safer tool use in LLM-based agents.

摘要: AI智能体是将大语言模型（LLM）与外部工具结合的自主系统，用于解决复杂任务。虽然此类工具扩展了能力，但不恰当的工具权限会带来间接提示注入和工具滥用等安全风险。我们将这些故障特征归纳为不平衡的工具驱动代理性：智能体可能保留不必要的权限（代理过度）或未能调用必需工具（代理不足），从而扩大攻击面并降低性能。本文提出AgenTRIM框架，该框架可在不改变智能体内部推理机制的情况下，检测并缓解工具驱动代理性风险。AgenTRIM通过互补的离线和在线阶段应对这些风险：离线阶段通过代码和执行轨迹重建并验证智能体的工具接口；运行时通过自适应过滤和状态感知的工具调用验证，实施基于步骤的最小权限工具访问。在AgentDojo基准测试中，AgenTRIM在保持高任务性能的同时显著降低了攻击成功率。附加实验表明其对基于描述的攻击具有鲁棒性，并能有效执行明确的安全策略。这些结果共同证明，AgenTRIM为基于LLM的智能体提供了一种实用且保持能力的安全工具使用方案。



## **17. Be Your Own Red Teamer: Safety Alignment via Self-Play and Reflective Experience Replay**

成为你自己的红队：通过自我博弈与反思经验回放实现安全对齐 cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10589v1) [paper-pdf](https://arxiv.org/pdf/2601.10589v1)

**Confidence**: 0.95

**Authors**: Hao Wang, Yanting Wang, Hao Li, Rui Li, Lei Sha

**Abstract**: Large Language Models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial ``jailbreak'' attacks designed to bypass safety guardrails. Current safety alignment methods depend heavily on static external red teaming, utilizing fixed defense prompts or pre-collected adversarial datasets. This leads to a rigid defense that overfits known patterns and fails to generalize to novel, sophisticated threats. To address this critical limitation, we propose empowering the model to be its own red teamer, capable of achieving autonomous and evolving adversarial attacks. Specifically, we introduce Safety Self- Play (SSP), a system that utilizes a single LLM to act concurrently as both the Attacker (generating jailbreaks) and the Defender (refusing harmful requests) within a unified Reinforcement Learning (RL) loop, dynamically evolving attack strategies to uncover vulnerabilities while simultaneously strengthening defense mechanisms. To ensure the Defender effectively addresses critical safety issues during the self-play, we introduce an advanced Reflective Experience Replay Mechanism, which uses an experience pool accumulated throughout the process. The mechanism employs a Upper Confidence Bound (UCB) sampling strategy to focus on failure cases with low rewards, helping the model learn from past hard mistakes while balancing exploration and exploitation. Extensive experiments demonstrate that our SSP approach autonomously evolves robust defense capabilities, significantly outperforming baselines trained on static adversarial datasets and establishing a new benchmark for proactive safety alignment.

摘要: 大语言模型（LLMs）已展现出卓越能力，但仍易受旨在绕过安全防护的对抗性“越狱”攻击。当前的安全对齐方法严重依赖静态外部红队测试，使用固定的防御提示或预收集的对抗数据集，导致防御僵化、过度拟合已知模式，难以泛化至新颖复杂的威胁。为克服这一关键局限，我们提出让模型成为自身的红队测试者，实现自主演进的对抗攻击。具体而言，我们引入安全自我博弈（SSP）系统，利用单一LLM在统一强化学习（RL）循环中同时扮演攻击者（生成越狱指令）和防御者（拒绝有害请求）角色，动态演进攻击策略以发现漏洞，同时强化防御机制。为确保防御者在自我博弈中有效处理关键安全问题，我们引入先进的反思经验回放机制，利用过程中积累的经验池，采用上置信界（UCB）采样策略聚焦低奖励的失败案例，帮助模型从过往困难错误中学习，并平衡探索与利用。大量实验表明，我们的SSP方法能自主演进稳健的防御能力，显著优于基于静态对抗数据集训练的基线方法，为主动安全对齐树立了新标杆。



## **18. The Straight and Narrow: Do LLMs Possess an Internal Moral Path?**

笔直而狭窄：大型语言模型是否具备内在道德路径？ cs.CL

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10307v1) [paper-pdf](https://arxiv.org/pdf/2601.10307v1)

**Confidence**: 0.95

**Authors**: Luoming Hu, Jingjie Zeng, Liang Yang, Hongfei Lin

**Abstract**: Enhancing the moral alignment of Large Language Models (LLMs) is a critical challenge in AI safety. Current alignment techniques often act as superficial guardrails, leaving the intrinsic moral representations of LLMs largely untouched. In this paper, we bridge this gap by leveraging Moral Foundations Theory (MFT) to map and manipulate the fine-grained moral landscape of LLMs. Through cross-lingual linear probing, we validate the shared nature of moral representations in middle layers and uncover a shared yet different moral subspace between English and Chinese. Building upon this, we extract steerable Moral Vectors and successfully validate their efficacy at both internal and behavioral levels. Leveraging the high generalizability of morality, we propose Adaptive Moral Fusion (AMF), a dynamic inference-time intervention that synergizes probe detection with vector injection to tackle the safety-helpfulness trade-off. Empirical results confirm that our approach acts as a targeted intrinsic defense, effectively reducing incorrect refusals on benign queries while minimizing jailbreak success rates compared to standard baselines.

摘要: 增强大型语言模型（LLMs）的道德对齐是AI安全领域的关键挑战。当前的对齐技术往往仅作为表层护栏，未能触及LLMs的内在道德表征。本文通过运用道德基础理论（MFT），对LLMs的细粒度道德图景进行映射与调控，以弥合这一差距。通过跨语言线性探测，我们验证了中间层道德表征的共享特性，并揭示了英语与中文之间共享但存在差异的道德子空间。基于此，我们提取了可引导的道德向量，并在内部表征与行为层面成功验证了其有效性。利用道德的高泛化性，我们提出了自适应道德融合（AMF）——一种动态推理时干预方法，通过协同探测检测与向量注入来应对安全性与有用性之间的权衡。实证结果表明，相较于标准基线，我们的方法能作为精准的内在防御机制，有效减少对良性查询的错误拒绝，同时显著降低越狱成功率。



## **19. Reasoning Hijacking: Subverting LLM Classification via Decision-Criteria Injection**

推理劫持：通过决策标准注入颠覆LLM分类 cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10294v1) [paper-pdf](https://arxiv.org/pdf/2601.10294v1)

**Confidence**: 0.95

**Authors**: Yuansen Liu, Yixuan Tang, Anthony Kum Hoe Tun

**Abstract**: Current LLM safety research predominantly focuses on mitigating Goal Hijacking, preventing attackers from redirecting a model's high-level objective (e.g., from "summarizing emails" to "phishing users"). In this paper, we argue that this perspective is incomplete and highlight a critical vulnerability in Reasoning Alignment. We propose a new adversarial paradigm: Reasoning Hijacking and instantiate it with Criteria Attack, which subverts model judgments by injecting spurious decision criteria without altering the high-level task goal. Unlike Goal Hijacking, which attempts to override the system prompt, Reasoning Hijacking accepts the high-level goal but manipulates the model's decision-making logic by injecting spurious reasoning shortcut. Though extensive experiments on three different tasks (toxic comment, negative review, and spam detection), we demonstrate that even newest models are prone to prioritize injected heuristic shortcuts over rigorous semantic analysis. The results are consistent over different backbones. Crucially, because the model's "intent" remains aligned with the user's instructions, these attacks can bypass defenses designed to detect goal deviation (e.g., SecAlign, StruQ), exposing a fundamental blind spot in the current safety landscape. Data and code are available at https://github.com/Yuan-Hou/criteria_attack

摘要: 当前LLM安全研究主要聚焦于缓解目标劫持，防止攻击者重定向模型的高级目标（例如从“总结邮件”转向“网络钓鱼用户”）。本文认为这一视角并不完整，并揭示了推理对齐中的关键漏洞。我们提出一种新的对抗范式：推理劫持，并通过标准攻击进行实例化——该方法通过注入虚假决策标准来颠覆模型判断，同时不改变高级任务目标。与试图覆盖系统提示的目标劫持不同，推理劫持接受高级目标，但通过注入虚假推理捷径来操纵模型的决策逻辑。通过对三项不同任务（有害评论、负面评论和垃圾信息检测）的广泛实验，我们证明即使最新模型也倾向于优先采用注入的启发式捷径而非严谨的语义分析。该结果在不同骨干模型上保持一致。关键在于，由于模型的“意图”仍与用户指令保持一致，这类攻击能够绕过旨在检测目标偏离的防御机制（如SecAlign、StruQ），暴露出当前安全体系中的根本性盲点。数据与代码详见https://github.com/Yuan-Hou/criteria_attack



## **20. ReasAlign: Reasoning Enhanced Safety Alignment against Prompt Injection Attack**

ReasAlign：针对提示注入攻击的推理增强型安全对齐方法 cs.CR

15 pages, 10 figures

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10173v1) [paper-pdf](https://arxiv.org/pdf/2601.10173v1)

**Confidence**: 0.95

**Authors**: Hao Li, Yankai Yang, G. Edward Suh, Ning Zhang, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) have enabled the development of powerful agentic systems capable of automating complex workflows across various fields. However, these systems are highly vulnerable to indirect prompt injection attacks, where malicious instructions embedded in external data can hijack agent behavior. In this work, we present ReasAlign, a model-level solution to improve safety alignment against indirect prompt injection attacks. The core idea of ReasAlign is to incorporate structured reasoning steps to analyze user queries, detect conflicting instructions, and preserve the continuity of the user's intended tasks to defend against indirect injection attacks. To further ensure reasoning logic and accuracy, we introduce a test-time scaling mechanism with a preference-optimized judge model that scores reasoning steps and selects the best trajectory. Comprehensive evaluations across various benchmarks show that ReasAlign maintains utility comparable to an undefended model while consistently outperforming Meta SecAlign, the strongest prior guardrail. On the representative open-ended CyberSecEval2 benchmark, which includes multiple prompt-injected tasks, ReasAlign achieves 94.6% utility and only 3.6% ASR, far surpassing the state-of-the-art defensive model of Meta SecAlign (56.4% utility and 74.4% ASR). These results demonstrate that ReasAlign achieves the best trade-off between security and utility, establishing a robust and practical defense against prompt injection attacks in real-world agentic systems. Our code and experimental results could be found at https://github.com/leolee99/ReasAlign.

摘要: 大型语言模型（LLMs）推动了强大智能体系统的发展，使其能够自动化各领域的复杂工作流程。然而，这些系统极易受到间接提示注入攻击，即外部数据中嵌入的恶意指令可能劫持智能体行为。本文提出ReasAlign，一种模型级解决方案，旨在增强针对间接提示注入攻击的安全对齐能力。ReasAlign的核心思想是通过结构化推理步骤来分析用户查询、检测冲突指令，并保持用户预期任务的连续性以抵御间接注入攻击。为确保推理逻辑与准确性，我们引入测试时扩展机制，采用偏好优化的评判模型对推理步骤进行评分并选择最优轨迹。跨多个基准的综合评估表明，ReasAlign在保持与未防御模型相当实用性的同时，始终优于现有最强防护框架Meta SecAlign。在包含多重提示注入任务的代表性开放式基准CyberSecEval2上，ReasAlign实现94.6%的实用性和仅3.6%的攻击成功率（ASR），显著超越当前最优防御模型Meta SecAlign（56.4%实用性和74.4% ASR）。这些结果表明ReasAlign在安全性与实用性之间实现了最佳平衡，为现实世界智能体系统建立了针对提示注入攻击的鲁棒且实用的防御方案。代码与实验结果详见https://github.com/leolee99/ReasAlign。



## **21. ToolSafe: Enhancing Tool Invocation Safety of LLM-based agents via Proactive Step-level Guardrail and Feedback**

ToolSafe：通过主动式步骤级护栏与反馈增强基于LLM的智能体工具调用安全性 cs.CL

Work in Progress. Code available: https://github.com/MurrayTom/ToolSafe

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10156v1) [paper-pdf](https://arxiv.org/pdf/2601.10156v1)

**Confidence**: 0.95

**Authors**: Yutao Mou, Zhangchi Xue, Lijun Li, Peiyang Liu, Shikun Zhang, Wei Ye, Jing Shao

**Abstract**: While LLM-based agents can interact with environments via invoking external tools, their expanded capabilities also amplify security risks. Monitoring step-level tool invocation behaviors in real time and proactively intervening before unsafe execution is critical for agent deployment, yet remains under-explored. In this work, we first construct TS-Bench, a novel benchmark for step-level tool invocation safety detection in LLM agents. We then develop a guardrail model, TS-Guard, using multi-task reinforcement learning. The model proactively detects unsafe tool invocation actions before execution by reasoning over the interaction history. It assesses request harmfulness and action-attack correlations, producing interpretable and generalizable safety judgments and feedback. Furthermore, we introduce TS-Flow, a guardrail-feedback-driven reasoning framework for LLM agents, which reduces harmful tool invocations of ReAct-style agents by 65 percent on average and improves benign task completion by approximately 10 percent under prompt injection attacks.

摘要: 尽管基于大语言模型（LLM）的智能体能够通过调用外部工具与环境交互，但其扩展的能力也同时放大了安全风险。实时监控步骤级工具调用行为并在不安全执行前主动干预，对于智能体部署至关重要，但目前仍缺乏深入研究。本研究首先构建了TS-Bench——一个面向LLM智能体步骤级工具调用安全检测的新型基准。随后，我们利用多任务强化学习开发了护栏模型TS-Guard。该模型通过推理交互历史，在执行前主动检测不安全的工具调用行为，评估请求危害性及行为-攻击关联性，生成可解释且可泛化的安全判断与反馈。此外，我们提出了TS-Flow——一个护栏反馈驱动的LLM智能体推理框架，该框架在提示注入攻击下，平均将ReAct式智能体的有害工具调用减少65%，并将良性任务完成率提升约10%。



## **22. Understanding and Preserving Safety in Fine-Tuned LLMs**

理解与保持微调后大语言模型的安全性 cs.LG

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10141v1) [paper-pdf](https://arxiv.org/pdf/2601.10141v1)

**Confidence**: 0.95

**Authors**: Jiawen Zhang, Yangfan Hu, Kejia Chen, Lipeng He, Jiachen Ma, Jian Lou, Dan Li, Jian Liu, Xiaohu Yang, Ruoxi Jia

**Abstract**: Fine-tuning is an essential and pervasive functionality for applying large language models (LLMs) to downstream tasks. However, it has the potential to substantially degrade safety alignment, e.g., by greatly increasing susceptibility to jailbreak attacks, even when the fine-tuning data is entirely harmless. Despite garnering growing attention in defense efforts during the fine-tuning stage, existing methods struggle with a persistent safety-utility dilemma: emphasizing safety compromises task performance, whereas prioritizing utility typically requires deep fine-tuning that inevitably leads to steep safety declination.   In this work, we address this dilemma by shedding new light on the geometric interaction between safety- and utility-oriented gradients in safety-aligned LLMs. Through systematic empirical analysis, we uncover three key insights: (I) safety gradients lie in a low-rank subspace, while utility gradients span a broader high-dimensional space; (II) these subspaces are often negatively correlated, causing directional conflicts during fine-tuning; and (III) the dominant safety direction can be efficiently estimated from a single sample. Building upon these novel insights, we propose safety-preserving fine-tuning (SPF), a lightweight approach that explicitly removes gradient components conflicting with the low-rank safety subspace. Theoretically, we show that SPF guarantees utility convergence while bounding safety drift. Empirically, SPF consistently maintains downstream task performance and recovers nearly all pre-trained safety alignment, even under adversarial fine-tuning scenarios. Furthermore, SPF exhibits robust resistance to both deep fine-tuning and dynamic jailbreak attacks. Together, our findings provide new mechanistic understanding and practical guidance toward always-aligned LLM fine-tuning.

摘要: 微调是将大语言模型应用于下游任务的关键且普遍的功能。然而，即使微调数据完全无害，它也可能显著降低安全对齐性，例如大幅增加对越狱攻击的易感性。尽管在微调阶段的防御工作中日益受到关注，现有方法仍面临持续的安全-效用困境：强调安全性会损害任务性能，而优先考虑效用通常需要深度微调，这不可避免地导致安全性急剧下降。在本工作中，我们通过重新审视安全对齐大语言模型中安全导向与效用导向梯度之间的几何相互作用来解决这一困境。通过系统性的实证分析，我们揭示了三个关键发现：（I）安全梯度位于低秩子空间，而效用梯度则跨越更广泛的高维空间；（II）这些子空间通常呈负相关，导致微调过程中的方向冲突；（III）主导安全方向可通过单一样本高效估计。基于这些新见解，我们提出了安全保持微调，这是一种轻量级方法，能显式移除与低秩安全子空间冲突的梯度分量。理论上，我们证明SPF能保证效用收敛同时限制安全漂移。实证表明，即使在对抗性微调场景下，SPF也能持续保持下游任务性能，并恢复几乎所有预训练安全对齐。此外，SPF对深度微调和动态越狱攻击均表现出稳健的抵抗能力。我们的研究结果为始终对齐的LLM微调提供了新的机制理解和实践指导。



## **23. YaPO: Learnable Sparse Activation Steering Vectors for Domain Adaptation**

YaPO：面向领域适应的可学习稀疏激活导向向量 cs.AI

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08441v1) [paper-pdf](https://arxiv.org/pdf/2601.08441v1)

**Confidence**: 0.95

**Authors**: Abdelaziz Bounhar, Rania Hossam Elmohamady Elbadry, Hadi Abdine, Preslav Nakov, Michalis Vazirgiannis, Guokan Shang

**Abstract**: Steering Large Language Models (LLMs) through activation interventions has emerged as a lightweight alternative to fine-tuning for alignment and personalization. Recent work on Bi-directional Preference Optimization (BiPO) shows that dense steering vectors can be learned directly from preference data in a Direct Preference Optimization (DPO) fashion, enabling control over truthfulness, hallucinations, and safety behaviors. However, dense steering vectors often entangle multiple latent factors due to neuron multi-semanticity, limiting their effectiveness and stability in fine-grained settings such as cultural alignment, where closely related values and behaviors (e.g., among Middle Eastern cultures) must be distinguished. In this paper, we propose Yet another Policy Optimization (YaPO), a \textit{reference-free} method that learns \textit{sparse steering vectors} in the latent space of a Sparse Autoencoder (SAE). By optimizing sparse codes, YaPO produces disentangled, interpretable, and efficient steering directions. Empirically, we show that YaPO converges faster, achieves stronger performance, and exhibits improved training stability compared to dense steering baselines. Beyond cultural alignment, YaPO generalizes to a range of alignment-related behaviors, including hallucination, wealth-seeking, jailbreak, and power-seeking. Importantly, YaPO preserves general knowledge, with no measurable degradation on MMLU. Overall, our results show that YaPO provides a general recipe for efficient, stable, and fine-grained alignment of LLMs, with broad applications to controllability and domain adaptation. The associated code and data are publicly available\footnote{https://github.com/MBZUAI-Paris/YaPO}.

摘要: 通过激活干预引导大语言模型已成为一种轻量级的对齐与个性化替代方案，无需微调。最近的双向偏好优化研究表明，可以直接从偏好数据中以直接偏好优化的方式学习密集导向向量，从而实现对真实性、幻觉和安全行为的控制。然而，由于神经元的多元语义性，密集导向向量常常纠缠多个潜在因素，限制了其在细粒度场景（如文化对齐）中的有效性和稳定性，其中必须区分密切相关的价值观和行为（例如中东文化之间）。本文提出另一种策略优化方法，这是一种在稀疏自编码器潜在空间中学习稀疏导向向量的无参考方法。通过优化稀疏编码，YaPO产生解耦、可解释且高效的导向方向。实验表明，与密集导向基线相比，YaPO收敛更快、性能更强且训练稳定性更佳。除文化对齐外，YaPO可泛化至一系列对齐相关行为，包括幻觉、财富追求、越狱和权力追求。重要的是，YaPO能保持通用知识，在MMLU基准上未见性能下降。总体而言，我们的结果表明，YaPO为LLMs的高效、稳定和细粒度对齐提供了通用方案，在可控性和领域适应方面具有广泛应用前景。相关代码和数据已公开。



## **24. What Matters For Safety Alignment?**

安全对齐的关键因素探究 cs.CL

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.03868v1) [paper-pdf](https://arxiv.org/pdf/2601.03868v1)

**Confidence**: 0.95

**Authors**: Xing Li, Hui-Ling Zhen, Lihao Yin, Xianzhi Yu, Zhenhua Dong, Mingxuan Yuan

**Abstract**: This paper presents a comprehensive empirical study on the safety alignment capabilities. We evaluate what matters for safety alignment in LLMs and LRMs to provide essential insights for developing more secure and reliable AI systems. We systematically investigate and compare the influence of six critical intrinsic model characteristics and three external attack techniques. Our large-scale evaluation is conducted using 32 recent, popular LLMs and LRMs across thirteen distinct model families, spanning a parameter scale from 3B to 235B. The assessment leverages five established safety datasets and probes model vulnerabilities with 56 jailbreak techniques and four CoT attack strategies, resulting in 4.6M API calls. Our key empirical findings are fourfold. First, we identify the LRMs GPT-OSS-20B, Qwen3-Next-80B-A3B-Thinking, and GPT-OSS-120B as the top-three safest models, which substantiates the significant advantage of integrated reasoning and self-reflection mechanisms for robust safety alignment. Second, post-training and knowledge distillation may lead to a systematic degradation of safety alignment. We thus argue that safety must be treated as an explicit constraint or a core optimization objective during these stages, not merely subordinated to the pursuit of general capability. Third, we reveal a pronounced vulnerability: employing a CoT attack via a response prefix can elevate the attack success rate by 3.34x on average and from 0.6% to 96.3% for Seed-OSS-36B-Instruct. This critical finding underscores the safety risks inherent in text-completion interfaces and features that allow user-defined response prefixes in LLM services, highlighting an urgent need for architectural and deployment safeguards. Fourth, roleplay, prompt injection, and gradient-based search for adversarial prompts are the predominant methodologies for eliciting unaligned behaviors in modern models.

摘要: 本文对安全对齐能力进行了全面的实证研究。我们评估了影响LLMs和LRMs安全对齐的关键因素，为开发更安全可靠的AI系统提供重要见解。我们系统性地研究并比较了六个关键内在模型特性和三种外部攻击技术的影响。我们的大规模评估涵盖了32个近期流行的LLMs和LRMs，涉及十三个不同的模型系列，参数规模从3B到235B。评估采用五个成熟的安全数据集，并通过56种越狱技术和四种CoT攻击策略探测模型漏洞，共计进行了460万次API调用。我们的核心实证发现包括四个方面：首先，我们确定LRMs模型GPT-OSS-20B、Qwen3-Next-80B-A3B-Thinking和GPT-OSS-120B为最安全的三个模型，这证实了集成推理和自我反思机制对稳健安全对齐的重要优势。其次，后训练和知识蒸馏可能导致安全对齐的系统性退化。因此我们认为，在这些阶段必须将安全视为明确的约束或核心优化目标，而不仅仅是追求通用能力的附属品。第三，我们揭示了一个显著漏洞：通过响应前缀实施CoT攻击，平均可将攻击成功率提升3.34倍，对于Seed-OSS-36B-Instruct模型甚至能从0.6%提升至96.3%。这一关键发现凸显了文本补全接口以及允许用户自定义响应前缀的LLM服务中固有的安全风险，亟需架构和部署层面的防护措施。第四，角色扮演、提示注入和基于梯度的对抗性提示搜索是现代模型中诱发未对齐行为的主要方法。



## **25. STAR-S: Improving Safety Alignment through Self-Taught Reasoning on Safety Rules**

STAR-S：通过基于安全规则的自学推理提升安全对齐 cs.AI

19 pages,4 figures

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.03537v1) [paper-pdf](https://arxiv.org/pdf/2601.03537v1)

**Confidence**: 0.95

**Authors**: Di Wu, Yanyan Zhao, Xin Lu, Mingzhe Li, Bing Qin

**Abstract**: Defending against jailbreak attacks is crucial for the safe deployment of Large Language Models (LLMs). Recent research has attempted to improve safety by training models to reason over safety rules before responding. However, a key issue lies in determining what form of safety reasoning effectively defends against jailbreak attacks, which is difficult to explicitly design or directly obtain. To address this, we propose \textbf{STAR-S} (\textbf{S}elf-\textbf{TA}ught \textbf{R}easoning based on \textbf{S}afety rules), a framework that integrates the learning of safety rule reasoning into a self-taught loop. The core of STAR-S involves eliciting reasoning and reflection guided by safety rules, then leveraging fine-tuning to enhance safety reasoning. Repeating this process creates a synergistic cycle. Improvements in the model's reasoning and interpretation of safety rules allow it to produce better reasoning data under safety rule prompts, which is then utilized for further training. Experiments show that STAR-S effectively defends against jailbreak attacks, outperforming baselines. Code is available at: https://github.com/pikepokenew/STAR_S.git.

摘要: 防御越狱攻击对于大语言模型（LLMs）的安全部署至关重要。近期研究尝试通过训练模型在响应前对安全规则进行推理来提升安全性。然而，关键问题在于确定何种形式的安全推理能有效防御越狱攻击，这难以显式设计或直接获取。为此，我们提出\textbf{STAR-S}（基于安全规则的\textbf{自}学\textbf{推}理框架），该框架将安全规则推理的学习整合进一个自学循环中。STAR-S的核心在于：基于安全规则引导生成推理与反思，进而通过微调强化安全推理能力。重复此过程可形成协同增强循环——模型对安全规则的理解与推理能力提升后，能在安全规则提示下生成更优质的推理数据，进而用于后续训练。实验表明，STAR-S能有效防御越狱攻击，性能优于基线方法。代码已开源：https://github.com/pikepokenew/STAR_S.git。



## **26. PromptScreen: Efficient Jailbreak Mitigation Using Semantic Linear Classification in a Multi-Staged Pipeline**

PromptScreen：基于语义线性分类的多阶段流程高效越狱缓解系统 cs.CR

Under Review

**SubmitDate**: 2026-01-09    [abs](http://arxiv.org/abs/2512.19011v2) [paper-pdf](https://arxiv.org/pdf/2512.19011v2)

**Confidence**: 0.95

**Authors**: Akshaj Prashanth Rao, Advait Singh, Saumya Kumaar Saksena, Dhruv Kumar

**Abstract**: Prompt injection and jailbreaking attacks pose persistent security challenges to large language model (LLM)-based systems. We present PromptScreen, an efficient and systematically evaluated defense architecture that mitigates these threats through a lightweight, multi-stage pipeline. Its core component is a semantic filter based on text normalization, TF-IDF representations, and a Linear SVM classifier. Despite its simplicity, this module achieves 93.4% accuracy and 96.5% specificity on held-out data, substantially reducing attack throughput while incurring negligible computational overhead.   Building on this efficient foundation, the full pipeline integrates complementary detection and mitigation mechanisms that operate at successive stages, providing strong robustness with minimal latency. In comparative experiments, our SVM-based configuration improves overall accuracy from 35.1% to 93.4% while reducing average time-to-completion from approximately 450 s to 47 s, yielding over 10 times lower latency than ShieldGemma. These results demonstrate that the proposed design simultaneously advances defensive precision and efficiency, addressing a core limitation of current model-based moderators.   Evaluation across a curated corpus of over 30,000 labeled prompts, including benign, jailbreak, and application-layer injections, confirms that staged, resource-efficient defenses can robustly secure modern LLM-driven applications.

摘要: 提示注入和越狱攻击对基于大语言模型（LLM）的系统构成持续的安全挑战。我们提出PromptScreen，一种经过系统评估的高效防御架构，通过轻量级多阶段流程缓解这些威胁。其核心组件是基于文本归一化、TF-IDF表示和线性SVM分类器的语义过滤器。尽管设计简洁，该模块在保留数据上实现了93.4%的准确率和96.5%的特异性，显著降低攻击吞吐量的同时带来可忽略的计算开销。在此高效基础上，完整流程集成了在连续阶段运行的互补检测与缓解机制，以最小延迟提供强鲁棒性。对比实验中，我们基于SVM的配置将整体准确率从35.1%提升至93.4%，同时将平均完成时间从约450秒降至47秒，延迟比ShieldGemma降低10倍以上。这些结果表明，所提设计在提升防御精度与效率方面取得同步进展，解决了当前基于模型的审核器的核心局限。通过对包含良性、越狱和应用层注入的超过30,000条标注提示的精选语料库进行评估，证实了分阶段、资源高效的防御机制能够为现代LLM驱动应用提供可靠安全保障。



## **27. MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Domain Risks in LLMs**

MENTOR：一种元认知驱动的自进化框架，用于发现和缓解大语言模型中的隐式领域风险 cs.AI

**SubmitDate**: 2026-01-08    [abs](http://arxiv.org/abs/2511.07107v2) [paper-pdf](https://arxiv.org/pdf/2511.07107v2)

**Confidence**: 0.95

**Authors**: Liang Shan, Kaicheng Shen, Wen Wu, Zhenyu Ying, Chaochao Lu, Yan Teng, Jingqi Huang, Guangze Ye, Guoqing Wang, Liang He

**Abstract**: Ensuring the safety of Large Language Models (LLMs) is critical for real-world deployment. However, current safety measures often fail to address implicit, domain-specific risks. To investigate this gap, we introduce a dataset of 3,000 annotated queries spanning education, finance, and management. Evaluations across 14 leading LLMs reveal a concerning vulnerability: an average jailbreak success rate of 57.8%. In response, we propose MENTOR, a metacognition-driven self-evolution framework. MENTOR first performs structured self-assessment through simulated critical thinking, such as perspective-taking and consequential reasoning to uncover latent model misalignments. These reflections are formalized into dynamic rule-based knowledge graphs that evolve with emerging risk patterns. To enforce these rules at inference time, we introduce activation steering, a method that directly modulates the model's internal representations to ensure compliance. Experiments demonstrate that MENTOR substantially reduces attack success rates across all tested domains and achieves risk analysis performance comparable to human experts. Our work offers a scalable and adaptive pathway toward robust domain-specific alignment of LLMs.

摘要: 确保大语言模型（LLMs）的安全对于实际部署至关重要。然而，当前的安全措施往往未能解决隐式的、特定领域的风险。为探究这一差距，我们构建了一个包含教育、金融和管理领域共3,000条标注查询的数据集。对14个主流LLMs的评估揭示了一个令人担忧的脆弱性：平均越狱成功率高达57.8%。为此，我们提出了MENTOR，一种元认知驱动的自进化框架。MENTOR首先通过模拟批判性思维（如换位思考和后果推理）进行结构化自我评估，以发现潜在的模型未对齐问题。这些反思被形式化为动态的基于规则的知识图谱，能够随新兴风险模式而进化。为了在推理时强制执行这些规则，我们引入了激活引导方法，该方法直接调节模型的内部表征以确保合规性。实验表明，MENTOR显著降低了所有测试领域的攻击成功率，并实现了与人类专家相当的风险分析性能。我们的工作为LLMs的鲁棒领域特定对齐提供了一条可扩展且自适应的路径。



## **28. Exploring the Secondary Risks of Large Language Models**

探索大型语言模型的次级风险 cs.LG

18 pages, 5 figures

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2506.12382v4) [paper-pdf](https://arxiv.org/pdf/2506.12382v4)

**Confidence**: 0.95

**Authors**: Jiawei Chen, Zhengwei Fang, Xiao Yang, Chao Yu, Zhaoxia Yin, Hang Su

**Abstract**: Ensuring the safety and alignment of Large Language Models is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives verbose response and speculative advice that capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. To support reproducible evaluation, we release SecRiskBench, a benchmark dataset of 650 prompts covering eight diverse real-world risk categories. Experimental results from extensive evaluations on 16 popular models demonstrate that secondary risks are widespread, transferable across models, and modality independent, emphasizing the urgent need for enhanced safety mechanisms to address benign yet harmful LLM behaviors in real-world deployments.

摘要: 随着大型语言模型日益融入关键应用和社会功能，确保其安全性和对齐性成为重大挑战。现有研究主要集中于越狱攻击，而对良性交互中微妙出现的非对抗性故障关注不足。我们提出次级风险这一新型故障模式，其特征是在良性提示下产生有害或误导性行为。与对抗攻击不同，这些风险源于不完美的泛化能力，且常能规避标准安全机制。为支持系统性评估，我们引入两个风险原语——冗长响应和推测性建议，以捕捉核心故障模式。基于这些定义，我们提出SecLens框架：一种通过优化任务相关性、风险激活度和语言合理性来有效引发次级风险行为的黑盒多目标搜索框架。为支持可复现评估，我们发布了SecRiskBench基准数据集，包含涵盖八个现实风险类别的650个提示。对16个主流模型的广泛实验表明，次级风险普遍存在、具有模型间可迁移性且与模态无关，这凸显了在现实部署中亟需增强安全机制以应对良性但有害的LLM行为。



## **29. PAM: Training Policy-Aligned Moderation Filters at Scale**

PAM：大规模训练策略对齐的内容审核过滤器 cs.CL

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2505.19766v3) [paper-pdf](https://arxiv.org/pdf/2505.19766v3)

**Confidence**: 0.95

**Authors**: Masoomali Fatehkia, Enes Altinisik, Mohamed Osman, Husrev Taha Sencar

**Abstract**: Large language models (LLMs) remain vulnerable to misalignment and jailbreaks, making external safeguards like moderation filters essential, yet existing filters often focus narrowly on safety, falling short of the broader alignment needs seen in real-world deployments. We introduce Policy Aligned Moderation (PAM), a flexible framework for training custom moderation filters grounded in user-defined policies that extend beyond conventional safety objectives. PAM automates training data generation without relying on human-written examples, enabling scalable support for diverse, application-specific alignment goals and generation policies. PAM-trained filters match the performance of state-of-the-art safety moderation filters and policy reasoning models, and outperform them on PAMbench, four newly introduced user-annotated policy enforcement benchmarks that target age restrictions, dietary accommodations, cultural alignment, and limitations in medical guidance. These performance gains are achieved while the PAM filter runs 5-100x faster at inference than policy-conditioned reasoning models.

摘要: 大型语言模型（LLMs）仍然容易受到错位和越狱攻击的影响，使得外部保障措施如内容审核过滤器变得至关重要。然而，现有过滤器通常仅聚焦于安全性，未能满足实际部署中更广泛的对齐需求。我们提出了策略对齐内容审核（PAM），这是一个灵活的框架，用于训练基于用户定义策略的定制化审核过滤器，这些策略超越了传统的安全目标。PAM无需依赖人工编写的示例即可自动生成训练数据，从而能够可扩展地支持多样化、特定于应用的对齐目标和生成策略。PAM训练的过滤器在性能上可与最先进的安全审核过滤器和策略推理模型相媲美，并在PAMbench（我们新引入的四个用户标注的策略执行基准测试，针对年龄限制、饮食适应、文化对齐和医疗指导限制）上表现更优。这些性能提升是在PAM过滤器推理速度比策略条件推理模型快5-100倍的情况下实现的。



## **30. STaR: Sensitive Trajectory Regulation for Unlearning in Large Reasoning Models**

STaR：面向大型推理模型遗忘学习的敏感轨迹调控 cs.AI

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09281v1) [paper-pdf](https://arxiv.org/pdf/2601.09281v1)

**Confidence**: 0.95

**Authors**: Jingjing Zhou, Gaoxiang Cong, Li Su, Liang Li

**Abstract**: Large Reasoning Models (LRMs) have advanced automated multi-step reasoning, but their ability to generate complex Chain-of-Thought (CoT) trajectories introduces severe privacy risks, as sensitive information may be deeply embedded throughout the reasoning process. Existing Large Language Models (LLMs) unlearning approaches that typically focus on modifying only final answers are insufficient for LRMs, as they fail to remove sensitive content from intermediate steps, leading to persistent privacy leakage and degraded security. To address these challenges, we propose Sensitive Trajectory Regulation (STaR), a parameter-free, inference-time unlearning framework that achieves robust privacy protection throughout the reasoning process. Specifically, we first identify sensitive content via semantic-aware detection. Then, we inject global safety constraints through secure prompt prefix. Next, we perform trajectory-aware suppression to dynamically block sensitive content across the entire reasoning chain. Finally, we apply token-level adaptive filtering to prevent both exact and paraphrased sensitive tokens during generation. Furthermore, to overcome the inadequacies of existing evaluation protocols, we introduce two metrics: Multi-Decoding Consistency Assessment (MCS), which measures the consistency of unlearning across diverse decoding strategies, and Multi-Granularity Membership Inference Attack (MIA) Evaluation, which quantifies privacy protection at both answer and reasoning-chain levels. Experiments on the R-TOFU benchmark demonstrate that STaR achieves comprehensive and stable unlearning with minimal utility loss, setting a new standard for privacy-preserving reasoning in LRMs.

摘要: 大型推理模型（LRMs）推动了自动化多步推理的发展，但其生成复杂思维链（CoT）轨迹的能力带来了严重的隐私风险——敏感信息可能深度嵌入整个推理过程。现有的大型语言模型（LLMs）遗忘学习方法通常仅关注修改最终答案，这对LRMs而言是不够的，因为它们无法从中间步骤中移除敏感内容，导致持续的隐私泄露和安全性下降。为应对这些挑战，我们提出了敏感轨迹调控（STaR），一种无需参数调整、在推理时执行的遗忘学习框架，可在整个推理过程中实现稳健的隐私保护。具体而言，我们首先通过语义感知检测识别敏感内容；接着通过安全提示前缀注入全局安全约束；然后执行轨迹感知抑制，动态阻断整个推理链中的敏感内容；最后应用词元级自适应过滤，在生成过程中防止精确及改写后的敏感词元出现。此外，为克服现有评估方案的不足，我们引入了两项指标：多解码一致性评估（MCS）——衡量不同解码策略下遗忘学习的一致性，以及多粒度成员推理攻击（MIA）评估——在答案和推理链两个层面量化隐私保护效果。在R-TOFU基准测试上的实验表明，STaR能以最小的效用损失实现全面稳定的遗忘学习，为LRMs的隐私保护推理设立了新标准。



## **31. SecureCAI: Injection-Resilient LLM Assistants for Cybersecurity Operations**

SecureCAI：面向网络安全操作的抗注入攻击LLM助手 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07835v1) [paper-pdf](https://arxiv.org/pdf/2601.07835v1)

**Confidence**: 0.95

**Authors**: Mohammed Himayath Ali, Mohammed Aqib Abdullah, Mohammed Mudassir Uddin, Shahnawaz Alam

**Abstract**: Large Language Models have emerged as transformative tools for Security Operations Centers, enabling automated log analysis, phishing triage, and malware explanation; however, deployment in adversarial cybersecurity environments exposes critical vulnerabilities to prompt injection attacks where malicious instructions embedded in security artifacts manipulate model behavior. This paper introduces SecureCAI, a novel defense framework extending Constitutional AI principles with security-aware guardrails, adaptive constitution evolution, and Direct Preference Optimization for unlearning unsafe response patterns, addressing the unique challenges of high-stakes security contexts where traditional safety mechanisms prove insufficient against sophisticated adversarial manipulation. Experimental evaluation demonstrates that SecureCAI reduces attack success rates by 94.7% compared to baseline models while maintaining 95.1% accuracy on benign security analysis tasks, with the framework incorporating continuous red-teaming feedback loops enabling dynamic adaptation to emerging attack strategies and achieving constitution adherence scores exceeding 0.92 under sustained adversarial pressure, thereby establishing a foundation for trustworthy integration of language model capabilities into operational cybersecurity workflows and addressing a critical gap in current approaches to AI safety within adversarial domains.

摘要: 大语言模型已成为安全运营中心的变革性工具，可实现自动化日志分析、钓鱼邮件分级和恶意软件解释；然而，在对抗性网络安全环境中的部署暴露了关键漏洞，即安全工件中嵌入的恶意指令可能通过提示注入攻击操纵模型行为。本文提出SecureCAI，这是一种新颖的防御框架，通过融合安全感知护栏、自适应宪法演进以及用于消除不安全响应模式的直接偏好优化，扩展了宪法AI原则，解决了高风险安全场景中的独特挑战——传统安全机制在面对复杂对抗性操纵时存在不足。实验评估表明，与基线模型相比，SecureCAI将攻击成功率降低了94.7%，同时在良性安全分析任务中保持95.1%的准确率；该框架通过持续红队反馈循环实现对新攻击策略的动态适应，在持续对抗压力下宪法遵循分数超过0.92，从而为语言模型能力可信集成到网络安全操作工作流奠定基础，并弥补了对抗性领域中当前AI安全方法的关键空白。



## **32. Defenses Against Prompt Attacks Learn Surface Heuristics**

针对提示攻击的防御机制仅习得表层启发式特征 cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07185v1) [paper-pdf](https://arxiv.org/pdf/2601.07185v1)

**Confidence**: 0.95

**Authors**: Shawn Li, Chenxiao Yu, Zhiyu Ni, Hao Li, Charith Peris, Chaowei Xiao, Yue Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in security-sensitive applications, where they must follow system- or developer-specified instructions that define the intended task behavior, while completing benign user requests. When adversarial instructions appear in user queries or externally retrieved content, models may override intended logic. Recent defenses rely on supervised fine-tuning with benign and malicious labels. Although these methods achieve high attack rejection rates, we find that they rely on narrow correlations in defense data rather than harmful intent, leading to systematic rejection of safe inputs. We analyze three recurring shortcut behaviors induced by defense fine-tuning. \emph{Position bias} arises when benign content placed later in a prompt is rejected at much higher rates; across reasoning benchmarks, suffix-task rejection rises from below \textbf{10\%} to as high as \textbf{90\%}. \emph{Token trigger bias} occurs when strings common in attack data raise rejection probability even in benign contexts; inserting a single trigger token increases false refusals by up to \textbf{50\%}. \emph{Topic generalization bias} reflects poor generalization beyond the defense data distribution, with defended models suffering test-time accuracy drops of up to \textbf{40\%}. These findings suggest that current prompt-injection defenses frequently respond to attack-like surface patterns rather than the underlying intent. We introduce controlled diagnostic datasets and a systematic evaluation across two base models and multiple defense pipelines, highlighting limitations of supervised fine-tuning for reliable LLM security.

摘要: 大型语言模型（LLMs）正日益部署于安全敏感的应用场景中，这些模型必须遵循系统或开发者定义的指令来确定预期任务行为，同时完成良性用户请求。当用户查询或外部检索内容中出现对抗性指令时，模型可能覆盖预设逻辑。现有防御机制主要依赖带有良性/恶意标签的监督微调。尽管这些方法能实现较高的攻击拒绝率，但我们发现其依赖的是防御数据中的局部相关性而非有害意图，导致系统性地拒绝安全输入。我们分析了防御微调引发的三种常见捷径行为：

*位置偏差*：当良性内容置于提示词后段时，拒绝率显著上升；在推理基准测试中，后缀任务的拒绝率从**10%以下**飙升至**90%**。

*词元触发偏差*：攻击数据中常见的字符串即使在良性语境下也会提高拒绝概率；插入单个触发词元可使错误拒绝率提升**50%**。

*主题泛化偏差*：模型在防御数据分布之外泛化能力不足，受保护模型在测试时的准确率下降高达**40%**。

这些发现表明，当前针对提示注入的防御机制往往仅对攻击性表层模式作出反应，而非识别深层意图。我们通过构建受控诊断数据集，并在两个基础模型和多种防御流程中进行系统评估，揭示了监督微调在实现可靠LLM安全方面的局限性。



## **33. VIGIL: Defending LLM Agents Against Tool Stream Injection via Verify-Before-Commit**

VIGIL：通过验证后提交机制防御LLM智能体工具流注入攻击 cs.CR

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.05755v2) [paper-pdf](https://arxiv.org/pdf/2601.05755v2)

**Confidence**: 0.95

**Authors**: Junda Lin, Zhaomeng Zhou, Zhi Zheng, Shuochen Liu, Tong Xu, Yong Chen, Enhong Chen

**Abstract**: LLM agents operating in open environments face escalating risks from indirect prompt injection, particularly within the tool stream where manipulated metadata and runtime feedback hijack execution flow. Existing defenses encounter a critical dilemma as advanced models prioritize injected rules due to strict alignment while static protection mechanisms sever the feedback loop required for adaptive reasoning. To reconcile this conflict, we propose \textbf{VIGIL}, a framework that shifts the paradigm from restrictive isolation to a verify-before-commit protocol. By facilitating speculative hypothesis generation and enforcing safety through intent-grounded verification, \textbf{VIGIL} preserves reasoning flexibility while ensuring robust control. We further introduce \textbf{SIREN}, a benchmark comprising 959 tool stream injection cases designed to simulate pervasive threats characterized by dynamic dependencies. Extensive experiments demonstrate that \textbf{VIGIL} outperforms state-of-the-art dynamic defenses by reducing the attack success rate by over 22\% while more than doubling the utility under attack compared to static baselines, thereby achieving an optimal balance between security and utility.

摘要: 在开放环境中运行的LLM智能体面临日益严重的间接提示注入风险，尤其是在工具流中，被篡改的元数据和运行时反馈会劫持执行流程。现有防御机制面临关键困境：先进模型因严格对齐而优先执行注入规则，而静态保护机制则切断了自适应推理所需的反馈循环。为解决这一矛盾，我们提出\textbf{VIGIL}框架，将防御范式从限制性隔离转向验证后提交协议。通过支持推测性假设生成并基于意图验证强制执行安全性，\textbf{VIGIL}在保持推理灵活性的同时确保鲁棒控制。我们进一步提出\textbf{SIREN}基准测试集，包含959个工具流注入案例，用于模拟具有动态依赖特征的普遍威胁。大量实验表明，\textbf{VIGIL}将攻击成功率降低超过22%，在受攻击时实用性较静态基线提升一倍以上，显著优于现有动态防御方案，从而实现了安全性与实用性的最优平衡。



## **34. Know Thy Enemy: Securing LLMs Against Prompt Injection via Diverse Data Synthesis and Instruction-Level Chain-of-Thought Learning**

知己知彼：通过多样化数据合成与指令级思维链学习保护大语言模型免受提示注入攻击 cs.AI

19 pages, 6 figures

**SubmitDate**: 2026-01-08    [abs](http://arxiv.org/abs/2601.04666v1) [paper-pdf](https://arxiv.org/pdf/2601.04666v1)

**Confidence**: 0.95

**Authors**: Zhiyuan Chang, Mingyang Li, Yuekai Huang, Ziyou Jiang, Xiaojun Jia, Qian Xiong, Junjie Wang, Zhaoyang Li, Qing Wang

**Abstract**: Large language model (LLM)-integrated applications have become increasingly prevalent, yet face critical security vulnerabilities from prompt injection (PI) attacks. Defending against PI attacks faces two major issues: malicious instructions can be injected through diverse vectors, and injected instructions often lack clear semantic boundaries from the surrounding context, making them difficult to identify. To address these issues, we propose InstruCoT, a model enhancement method for PI defense that synthesizes diverse training data and employs instruction-level chain-of-thought fine-tuning, enabling LLMs to effectively identify and reject malicious instructions regardless of their source or position in the context. We evaluate InstruCoT across three critical dimensions: Behavior Deviation, Privacy Leakage, and Harmful Output. Experimental results across four LLMs demonstrate that InstruCoT significantly outperforms baselines in all dimensions while maintaining utility performance without degradation

摘要: 大语言模型（LLM）集成应用日益普及，但面临提示注入（PI）攻击带来的严重安全漏洞。防御PI攻击存在两大挑战：恶意指令可通过多种途径注入，且注入指令常与上下文缺乏清晰的语义边界，难以识别。为解决这些问题，我们提出InstruCoT——一种用于PI防御的模型增强方法，通过合成多样化训练数据并采用指令级思维链微调，使LLM能够有效识别并拒绝恶意指令，无论其来源或上下文位置如何。我们从行为偏差、隐私泄露和有害输出三个关键维度评估InstruCoT。在四个LLM上的实验结果表明，InstruCoT在所有维度上均显著优于基线方法，同时保持实用性能不下降。



## **35. OFFSIDE: Benchmarking Unlearning Misinformation in Multimodal Large Language Models**

OFFSIDE：多模态大语言模型中虚假信息遗忘的基准测试 cs.AI

**SubmitDate**: 2026-01-03    [abs](http://arxiv.org/abs/2510.22535v2) [paper-pdf](https://arxiv.org/pdf/2510.22535v2)

**Confidence**: 0.95

**Authors**: Hao Zheng, Zirui Pang, Ling li, Zhijie Deng, Yuhan Pu, Zhaowei Zhu, Xiaobo Xia, Jiaheng Wei

**Abstract**: Advances in Multimodal Large Language Models (MLLMs) intensify concerns about data privacy, making Machine Unlearning (MU), the selective removal of learned information, a critical necessity. However, existing MU benchmarks for MLLMs are limited by a lack of image diversity, potential inaccuracies, and insufficient evaluation scenarios, which fail to capture the complexity of real-world applications. To facilitate the development of MLLMs unlearning and alleviate the aforementioned limitations, we introduce OFFSIDE, a novel benchmark for evaluating misinformation unlearning in MLLMs based on football transfer rumors. This manually curated dataset contains 15.68K records for 80 players, providing a comprehensive framework with four test sets to assess forgetting efficacy, generalization, utility, and robustness. OFFSIDE supports advanced settings like selective unlearning and corrective relearning, and crucially, unimodal unlearning (forgetting only text data). Our extensive evaluation of multiple baselines reveals key findings: (1) Unimodal methods (erasing text-based knowledge) fail on multimodal rumors; (2) Unlearning efficacy is largely driven by catastrophic forgetting; (3) All methods struggle with "visual rumors" (rumors appear in the image); (4) The unlearned rumors can be easily recovered and (5) All methods are vulnerable to prompt attacks. These results expose significant vulnerabilities in current approaches, highlighting the need for more robust multimodal unlearning solutions. The code is available at https://github.com/zh121800/OFFSIDE

摘要: 多模态大语言模型（MLLMs）的进展加剧了数据隐私的担忧，使得机器遗忘（MU）——选择性移除已学习信息——成为关键需求。然而，现有MLLMs的MU基准测试存在图像多样性不足、潜在不准确性及评估场景不充分等局限，难以反映实际应用的复杂性。为促进MLLMs遗忘能力的发展并缓解上述局限，我们提出了OFFSIDE——一个基于足球转会传闻的MLLMs虚假信息遗忘评估新基准。该人工标注数据集包含80名球员的15.68K条记录，通过四个测试集提供评估遗忘效能、泛化性、实用性和鲁棒性的综合框架。OFFSIDE支持选择性遗忘、纠正性再学习等高级设置，并关键性地支持单模态遗忘（仅遗忘文本数据）。我们对多种基线的广泛评估揭示了关键发现：（1）单模态方法（擦除基于文本的知识）对多模态传闻失效；（2）遗忘效能主要由灾难性遗忘驱动；（3）所有方法在处理'视觉传闻'（图像中包含传闻）时均存在困难；（4）已遗忘的传闻易被恢复；（5）所有方法均易受提示攻击。这些结果暴露了当前方法的显著脆弱性，凸显了开发更鲁棒的多模态遗忘解决方案的迫切需求。代码发布于https://github.com/zh121800/OFFSIDE



## **36. How to make Medical AI Systems safer? Simulating Vulnerabilities, and Threats in Multimodal Medical RAG System**

如何提升医疗AI系统的安全性？模拟多模态医疗RAG系统中的漏洞与威胁 cs.LG

Sumbitted to 2026 ICASSP

**SubmitDate**: 2026-01-04    [abs](http://arxiv.org/abs/2508.17215v2) [paper-pdf](https://arxiv.org/pdf/2508.17215v2)

**Confidence**: 0.95

**Authors**: Kaiwen Zuo, Zelin Liu, Raman Dutt, Ziyang Wang, Zhongtian Sun, Fan Mo, Pietro Liò

**Abstract**: Large Vision-Language Models (LVLMs) augmented with Retrieval-Augmented Generation (RAG) are increasingly employed in medical AI to enhance factual grounding through external clinical image-text retrieval. However, this reliance creates a significant attack surface. We propose MedThreatRAG, a novel multimodal poisoning framework that systematically probes vulnerabilities in medical RAG systems by injecting adversarial image-text pairs. A key innovation of our approach is the construction of a simulated semi-open attack environment, mimicking real-world medical systems that permit periodic knowledge base updates via user or pipeline contributions. Within this setting, we introduce and emphasize Cross-Modal Conflict Injection (CMCI), which embeds subtle semantic contradictions between medical images and their paired reports. These mismatches degrade retrieval and generation by disrupting cross-modal alignment while remaining sufficiently plausible to evade conventional filters. While basic textual and visual attacks are included for completeness, CMCI demonstrates the most severe degradation. Evaluations on IU-Xray and MIMIC-CXR QA tasks show that MedThreatRAG reduces answer F1 scores by up to 27.66% and lowers LLaVA-Med-1.5 F1 rates to as low as 51.36%. Our findings expose fundamental security gaps in clinical RAG systems and highlight the urgent need for threat-aware design and robust multimodal consistency checks. Finally, we conclude with a concise set of guidelines to inform the safe development of future multimodal medical RAG systems.

摘要: 结合检索增强生成（RAG）的大型视觉语言模型（LVLMs）在医疗AI领域日益普及，通过外部临床图文检索增强事实基础。然而，这种依赖性也带来了显著的攻击面。我们提出MedThreatRAG，一种新颖的多模态投毒框架，通过注入对抗性图文对，系统性地探测医疗RAG系统的漏洞。该方法的核心创新在于构建了一个模拟的半开放攻击环境，模仿允许通过用户或流程贡献定期更新知识库的真实医疗系统。在此环境中，我们引入并强调跨模态冲突注入（CMCI），即在医学图像与其配对报告之间嵌入细微的语义矛盾。这些不匹配通过破坏跨模态对齐而降低检索和生成质量，同时保持足够的合理性以规避常规过滤器。虽然为完整性包含了基本的文本和视觉攻击，但CMCI表现出最严重的性能退化。在IU-Xray和MIMIC-CXR QA任务上的评估显示，MedThreatRAG使答案F1分数降低高达27.66%，并将LLaVA-Med-1.5的F1率降至最低51.36%。我们的研究揭示了临床RAG系统的根本性安全漏洞，并强调了威胁感知设计和鲁棒多模态一致性检查的迫切需求。最后，我们总结了一套简明指南，为未来多模态医疗RAG系统的安全开发提供参考。



## **37. SLIP: Soft Label Mechanism and Key-Extraction-Guided CoT-based Defense Against Instruction Backdoor in APIs**

SLIP：基于软标签机制和关键词提取引导思维链的API指令后门防御方法 cs.CR

**SubmitDate**: 2026-01-05    [abs](http://arxiv.org/abs/2508.06153v2) [paper-pdf](https://arxiv.org/pdf/2508.06153v2)

**Confidence**: 0.95

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Haowei Chang, Yinghan Zhou, Yiming Xue

**Abstract**: With the development of customized large language model (LLM) agents, a new threat of black-box backdoor attacks has emerged, where malicious instructions are injected into hidden system prompts. These attacks easily bypass existing defenses that rely on white-box access, posing a serious security challenge. To address this, we propose SLIP, a Soft Label mechanism and key-extraction-guided CoT-based defense against Instruction backdoors in APIs. SLIP is designed based on two key insights. First, to counteract the model's oversensitivity to triggers, we propose a Key-extraction-guided Chain-of-Thought (KCoT). Instead of only considering the single trigger or the input sentence, KCoT prompts the agent to extract task-relevant key phrases. Second, to guide the LLM toward correct answers, our proposed Soft Label Mechanism (SLM) prompts the agent to quantify the semantic correlation between key phrases and candidate answers. Crucially, to mitigate the influence of residual triggers or misleading content in phrases extracted by KCoT, which typically causes anomalous scores, SLM excludes anomalous scores deviating significantly from the mean and subsequently averages the remaining scores to derive a more reliable semantic representation. Extensive experiments on classification and question-answer (QA) tasks demonstrate that SLIP is highly effective, reducing the average attack success rate (ASR) from 90.2% to 25.13% while maintaining high accuracy on clean data and outperforming state-of-the-art defenses. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/SLIP.

摘要: 随着定制化大语言模型（LLM）智能体的发展，一种新的黑盒后门攻击威胁已经出现，即恶意指令被注入到隐藏的系统提示中。这类攻击能够轻易绕过依赖白盒访问的现有防御机制，构成了严峻的安全挑战。为解决此问题，我们提出了SLIP，一种针对API中指令后门的基于软标签机制和关键词提取引导思维链的防御方法。SLIP的设计基于两个关键洞见：首先，为抵消模型对触发词的过度敏感，我们提出了关键词提取引导的思维链（KCoT）。KCoT不再仅考虑单一触发词或输入句子，而是提示智能体提取与任务相关的关键短语。其次，为引导LLM得出正确答案，我们提出的软标签机制（SLM）提示智能体量化关键短语与候选答案之间的语义相关性。至关重要的是，为减轻KCoT提取的短语中残留触发词或误导性内容的影响（这通常会导致异常分数），SLM会排除显著偏离平均值的异常分数，随后对剩余分数进行平均，以获得更可靠的语义表示。在分类和问答（QA）任务上的大量实验表明，SLIP非常有效，能将平均攻击成功率（ASR）从90.2%降至25.13%，同时在干净数据上保持高准确率，并优于最先进的防御方法。我们的代码可在 https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/SLIP 获取。



## **38. Text2VLM: Adapting Text-Only Datasets to Evaluate Alignment Training in Visual Language Models**

Text2VLM：适配纯文本数据集以评估视觉语言模型的对齐训练 cs.CL

9 pages, 9 figures. Jake Thomas served as Editor for this manuscript

**SubmitDate**: 2026-01-05    [abs](http://arxiv.org/abs/2507.20704v2) [paper-pdf](https://arxiv.org/pdf/2507.20704v2)

**Confidence**: 0.95

**Authors**: Gabriel Downer, Sean Craven, Damian Ruck, Jake Thomas

**Abstract**: The increasing integration of Visual Language Models (VLMs) into AI systems necessitates robust model alignment, especially when handling multimodal content that combines text and images. Existing evaluation datasets heavily lean towards text-only prompts, leaving visual vulnerabilities under evaluated. To address this gap, we propose \textbf{Text2VLM}, a novel multi-stage pipeline that adapts text-only datasets into multimodal formats, specifically designed to evaluate the resilience of VLMs against typographic prompt injection attacks. The Text2VLM pipeline identifies harmful content in the original text and converts it into a typographic image, creating a multimodal prompt for VLMs. Also, our evaluation of open-source VLMs highlights their increased susceptibility to prompt injection when visual inputs are introduced, revealing critical weaknesses in the current models' alignment. This is in addition to a significant performance gap compared to closed-source frontier models. We validate Text2VLM through human evaluations, ensuring the alignment of extracted salient concepts; text summarization and output classification align with human expectations. Text2VLM provides a scalable tool for comprehensive safety assessment, contributing to the development of more robust safety mechanisms for VLMs. By enhancing the evaluation of multimodal vulnerabilities, Text2VLM plays a role in advancing the safe deployment of VLMs in diverse, real-world applications.

摘要: 随着视觉语言模型（VLMs）日益融入AI系统，模型对齐的鲁棒性变得至关重要，尤其是在处理结合文本与图像的多模态内容时。现有的评估数据集严重偏向纯文本提示，导致视觉漏洞评估不足。为填补这一空白，我们提出\textbf{Text2VLM}——一种新颖的多阶段流程，可将纯文本数据集适配为多模态格式，专门用于评估VLMs抵抗排版提示注入攻击的韧性。Text2VLM流程首先识别原始文本中的有害内容，并将其转换为排版图像，从而为VLMs创建多模态提示。此外，我们对开源VLMs的评估表明，在引入视觉输入时，模型对提示注入的敏感性显著增加，揭示了当前模型对齐机制的关键弱点。与闭源前沿模型相比，这些模型还存在显著的性能差距。我们通过人工评估验证了Text2VLM的有效性，确保提取的关键概念对齐、文本摘要和输出分类均符合人类预期。Text2VLM为全面安全评估提供了可扩展工具，有助于开发更鲁棒的VLM安全机制。通过增强多模态漏洞的评估能力，Text2VLM在推进VLMs在多样化现实应用中的安全部署方面发挥着重要作用。



## **39. E$^2$AT: Multimodal Jailbreak Defense via Dynamic Joint Optimization for Multimodal Large Language Models**

E²AT：通过动态联合优化实现多模态大语言模型的多模态越狱防御 cs.CV

**SubmitDate**: 2026-01-06    [abs](http://arxiv.org/abs/2503.04833v3) [paper-pdf](https://arxiv.org/pdf/2503.04833v3)

**Confidence**: 0.95

**Authors**: Liming Lu, Xiang Gu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Xu Zheng, Yongbin Zhou

**Abstract**: Research endeavors have been made in learning robust Multimodal Large Language Models (MLLMs) against jailbreak attacks. However, existing methods for improving MLLMs' robustness still face critical challenges: \ding{172} how to efficiently tune massive weight parameters and \ding{173} how to ensure robustness against attacks across both visual and textual modalities. To this end, we propose an \textbf{E}fficient \textbf{E}nd-to-end \textbf{A}dversarial \textbf{T}raining (E$^2$AT) framework for both visual and textual adversarial attacks. Specifically, for the visual aspect, E$^2$AT incorporates an efficient projector-based AT module that aligns the attack samples at the feature level. For training objectives, we propose a Dynamic Joint Multimodal Optimization (DJMO) strategy to enhance generalization ability against jailbreak attacks by dynamically adjusting weights between normal and adversarial objectives. Extensive experiments are conducted with five major jailbreak attack methods across three mainstream MLLMs. Results demonstrate that our E$^2$AT achieves the state-of-the-art performance, outperforming existing baselines by an average margin of 34\% across text and image modalities, while maintaining clean task performance. Furthermore, evaluations of real-world embodied intelligent systems highlight the practical applicability of E$^2$AT, paving the way for the development of more secure and reliable multimodal systems. Our code is available on \href{https://anonymous.4open.science/r/E2AT_568}{\textcolor{red}{https://anonymous.4open.science/r/E2AT\_568}}.

摘要: 现有研究致力于学习鲁棒的多模态大语言模型（MLLMs）以抵御越狱攻击。然而，现有提升MLLMs鲁棒性的方法仍面临关键挑战：①如何高效微调海量权重参数；②如何确保对视觉和文本双模态攻击的鲁棒性。为此，我们提出一种面向视觉与文本对抗攻击的**高效端到端对抗训练**（E²AT）框架。具体而言，在视觉方面，E²AT引入基于高效投影器的对抗训练模块，在特征层面对齐攻击样本。在训练目标上，我们提出**动态联合多模态优化**（DJMO）策略，通过动态调整正常目标与对抗目标间的权重，增强模型对越狱攻击的泛化能力。我们在三种主流MLLMs上使用五种主要越狱攻击方法进行了广泛实验。结果表明，E²AT取得了最先进的性能，在文本和图像模态上平均超越现有基线34%，同时保持干净任务性能。此外，对现实世界具身智能系统的评估凸显了E²AT的实际适用性，为开发更安全可靠的多模态系统铺平了道路。代码发布于：https://anonymous.4open.science/r/E2AT_568。



## **40. PII-VisBench: Evaluating Personally Identifiable Information Safety in Vision Language Models Along a Continuum of Visibility**

PII-VisBench：沿可见度连续体评估视觉语言模型中的个人可识别信息安全 cs.AI

**SubmitDate**: 2026-01-09    [abs](http://arxiv.org/abs/2601.05739v1) [paper-pdf](https://arxiv.org/pdf/2601.05739v1)

**Confidence**: 0.95

**Authors**: G M Shahariar, Zabir Al Nazi, Md Olid Hasan Bhuiyan, Zhouxing Shi

**Abstract**: Vision Language Models (VLMs) are increasingly integrated into privacy-critical domains, yet existing evaluations of personally identifiable information (PII) leakage largely treat privacy as a static extraction task and ignore how a subject's online presence--the volume of their data available online--influences privacy alignment. We introduce PII-VisBench, a novel benchmark containing 4000 unique probes designed to evaluate VLM safety through the continuum of online presence. The benchmark stratifies 200 subjects into four visibility categories: high, medium, low, and zero--based on the extent and nature of their information available online. We evaluate 18 open-source VLMs (0.3B-32B) based on two key metrics: percentage of PII probing queries refused (Refusal Rate) and the fraction of non-refusal responses flagged for containing PII (Conditional PII Disclosure Rate). Across models, we observe a consistent pattern: refusals increase and PII disclosures decrease (9.10% high to 5.34% low) as subject visibility drops. We identify that models are more likely to disclose PII for high-visibility subjects, alongside substantial model-family heterogeneity and PII-type disparities. Finally, paraphrasing and jailbreak-style prompts expose attack and model-dependent failures, motivating visibility-aware safety evaluation and training interventions.

摘要: 视觉语言模型（VLMs）正日益融入隐私关键领域，然而现有对个人可识别信息（PII）泄露的评估大多将隐私视为静态提取任务，忽略了主体的在线存在（其在线可用数据量）如何影响隐私对齐。我们提出了PII-VisBench，这是一个包含4000个独特探针的新型基准，旨在通过在线存在的连续体评估VLM安全性。该基准根据200名主体在线信息的范围和性质，将其分为四个可见度类别：高、中、低和零。我们基于两个关键指标评估了18个开源VLM（0.3B-32B）：拒绝PII探测查询的百分比（拒绝率）以及被标记为包含PII的非拒绝响应比例（条件性PII披露率）。在所有模型中，我们观察到一致模式：随着主体可见度降低，拒绝率增加而PII披露减少（从高可见度的9.10%降至低可见度的5.34%）。我们发现模型更可能披露高可见度主体的PII，同时存在显著的模型系列异质性和PII类型差异。最后，释义和越狱式提示暴露了攻击和模型相关的失败，这推动了基于可见度的安全性评估和训练干预的需求。



## **41. Few Tokens Matter: Entropy Guided Attacks on Vision-Language Models**

少数关键标记：基于熵的视觉语言模型攻击方法 cs.CV

19 Pages,11 figures,8 tables

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.21815v1) [paper-pdf](https://arxiv.org/pdf/2512.21815v1)

**Confidence**: 0.95

**Authors**: Mengqi He, Xinyu Tian, Xin Shen, Jinhong Ni, Shu Zou, Zhaoyuan Yang, Jing Zhang

**Abstract**: Vision-language models (VLMs) achieve remarkable performance but remain vulnerable to adversarial attacks. Entropy, a measure of model uncertainty, is strongly correlated with the reliability of VLM. Prior entropy-based attacks maximize uncertainty at all decoding steps, implicitly assuming that every token contributes equally to generation instability. We show instead that a small fraction (about 20%) of high-entropy tokens, i.e., critical decision points in autoregressive generation, disproportionately governs output trajectories. By concentrating adversarial perturbations on these positions, we achieve semantic degradation comparable to global methods while using substantially smaller budgets. More importantly, across multiple representative VLMs, such selective attacks convert 35-49% of benign outputs into harmful ones, exposing a more critical safety risk. Remarkably, these vulnerable high-entropy forks recur across architecturally diverse VLMs, enabling feasible transferability (17-26% harmful rates on unseen targets). Motivated by these findings, we propose Entropy-bank Guided Adversarial attacks (EGA), which achieves competitive attack success rates (93-95%) alongside high harmful conversion, thereby revealing new weaknesses in current VLM safety mechanisms.

摘要: 视觉语言模型（VLM）虽性能卓越，但仍易受对抗攻击。熵作为模型不确定性的度量指标，与VLM的可靠性高度相关。现有基于熵的攻击方法在所有解码步骤中最大化不确定性，隐含假设每个标记对生成不稳定性的贡献均等。本文发现，实际上仅约20%的高熵标记（即自回归生成中的关键决策点）对输出轨迹具有决定性影响。通过将对抗扰动集中于这些位置，我们以显著更小的计算代价实现了与全局方法相当的语义退化效果。更重要的是，在多个代表性VLM中，此类选择性攻击可将35-49%的良性输出转化为有害输出，暴露出更严峻的安全风险。值得注意的是，这些脆弱的高熵决策分支在不同架构的VLM中反复出现，实现了可行的迁移攻击（对未见目标达到17-26%的有害转化率）。基于这些发现，我们提出熵库引导对抗攻击（EGA）方法，在实现竞争性攻击成功率（93-95%）的同时保持高有害转化率，从而揭示了当前VLM安全机制的新弱点。



## **42. GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs**

GateBreaker：针对混合专家大语言模型的网关引导攻击 cs.CR

Accepted by USENIX Security'26

**SubmitDate**: 2025-12-25    [abs](http://arxiv.org/abs/2512.21008v2) [paper-pdf](https://arxiv.org/pdf/2512.21008v2)

**Confidence**: 0.95

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Mixture-of-Experts (MoE) architectures have advanced the scaling of Large Language Models (LLMs) by activating only a sparse subset of parameters per input, enabling state-of-the-art performance with reduced computational cost. As these models are increasingly deployed in critical domains, understanding and strengthening their alignment mechanisms is essential to prevent harmful outputs. However, existing LLM safety research has focused almost exclusively on dense architectures, leaving the unique safety properties of MoEs largely unexamined. The modular, sparsely-activated design of MoEs suggests that safety mechanisms may operate differently than in dense models, raising questions about their robustness.   In this paper, we present GateBreaker, the first training-free, lightweight, and architecture-agnostic attack framework that compromises the safety alignment of modern MoE LLMs at inference time. GateBreaker operates in three stages: (i) gate-level profiling, which identifies safety experts disproportionately routed on harmful inputs, (ii) expert-level localization, which localizes the safety structure within safety experts, and (iii) targeted safety removal, which disables the identified safety structure to compromise the safety alignment. Our study shows that MoE safety concentrates within a small subset of neurons coordinated by sparse routing. Selective disabling of these neurons, approximately 3% of neurons in the targeted expert layers, significantly increases the averaged attack success rate (ASR) from 7.4% to 64.9% against the eight latest aligned MoE LLMs with limited utility degradation. These safety neurons transfer across models within the same family, raising ASR from 17.9% to 67.7% with one-shot transfer attack. Furthermore, GateBreaker generalizes to five MoE vision language models (VLMs) with 60.9% ASR on unsafe image inputs.

摘要: 混合专家（MoE）架构通过每个输入仅激活稀疏的参数子集，推动了大语言模型（LLM）的规模化发展，在降低计算成本的同时实现了最先进的性能。随着这些模型在关键领域日益广泛部署，理解并强化其对齐机制对于防止有害输出至关重要。然而，现有的大语言模型安全研究几乎完全集中于密集架构，MoE独特的安全特性在很大程度上尚未得到检验。MoE模块化、稀疏激活的设计表明，其安全机制的运作方式可能与密集模型不同，这引发了对其鲁棒性的质疑。本文提出GateBreaker，这是首个无需训练、轻量级且架构无关的攻击框架，可在推理时破坏现代MoE大语言模型的安全对齐。GateBreaker分三个阶段操作：（i）网关级分析，识别在有害输入上被过度路由的安全专家；（ii）专家级定位，定位安全专家内部的安全结构；（iii）针对性安全移除，禁用已识别的安全结构以破坏安全对齐。研究表明，MoE的安全机制集中在由稀疏路由协调的少量神经元子集中。选择性禁用目标专家层中约3%的神经元，可将对八种最新对齐MoE大语言模型的平均攻击成功率（ASR）从7.4%显著提升至64.9%，同时效用下降有限。这些安全神经元可在同一模型家族内跨模型迁移，通过单次迁移攻击将ASR从17.9%提升至67.7%。此外，GateBreaker可泛化至五种MoE视觉语言模型（VLM），在不安全图像输入上实现60.9%的ASR。



## **43. SafeMed-R1: Adversarial Reinforcement Learning for Generalizable and Robust Medical Reasoning in Vision-Language Models**

SafeMed-R1：基于对抗性强化学习的视觉语言模型通用鲁棒医疗推理框架 cs.AI

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19317v1) [paper-pdf](https://arxiv.org/pdf/2512.19317v1)

**Confidence**: 0.95

**Authors**: A. A. Gde Yogi Pramana, Jason Ray, Anthony Jaya, Michael Wijaya

**Abstract**: Vision--Language Models (VLMs) show significant promise for Medical Visual Question Answering (VQA), yet their deployment in clinical settings is hindered by severe vulnerability to adversarial attacks. Standard adversarial training, while effective for simpler tasks, often degrades both generalization performance and the quality of generated clinical reasoning. We introduce SafeMed-R1, a hybrid defense framework that ensures robust performance while preserving high-quality, interpretable medical reasoning. SafeMed-R1 employs a two-stage approach: at training time, we integrate Adversarial Training with Group Relative Policy Optimization (AT-GRPO) to explicitly robustify the reasoning process against worst-case perturbations; at inference time, we augment the model with Randomized Smoothing to provide certified $L_2$-norm robustness guarantees. We evaluate SafeMed-R1 on the OmniMedVQA benchmark across eight medical imaging modalities comprising over 88,000 samples. Our experiments reveal that standard fine-tuned VLMs, despite achieving 95\% accuracy on clean inputs, collapse to approximately 25\% under PGD attacks. In contrast, SafeMed-R1 maintains 84.45\% accuracy under the same adversarial conditions, representing a 59 percentage point improvement in robustness. Furthermore, we demonstrate that models trained with explicit chain-of-thought reasoning exhibit superior adversarial robustness compared to instruction-only variants, suggesting a synergy between interpretability and security in medical AI systems.

摘要: 视觉语言模型（VLMs）在医疗视觉问答（VQA）领域展现出巨大潜力，但其在临床环境中的部署因对对抗性攻击的高度脆弱性而受阻。标准的对抗训练虽然在简单任务中有效，却常常损害模型的泛化性能和生成临床推理的质量。我们提出SafeMed-R1——一种混合防御框架，在确保鲁棒性能的同时保持高质量、可解释的医疗推理。SafeMed-R1采用两阶段方法：训练阶段，我们通过结合对抗训练与组相对策略优化（AT-GRPO）来显式强化推理过程以抵御最坏情况扰动；推理阶段，我们采用随机平滑技术为模型提供经过认证的L2范数鲁棒性保证。我们在包含八种医学影像模态、超过88,000个样本的OmniMedVQA基准上评估SafeMed-R1。实验表明，标准微调的VLMs在干净输入上虽能达到95%的准确率，但在PGD攻击下会骤降至约25%。相比之下，SafeMed-R1在相同对抗条件下仍保持84.45%的准确率，鲁棒性提升达59个百分点。此外，我们发现采用显式思维链推理训练的模型比仅使用指令训练的变体具有更强的对抗鲁棒性，这揭示了医疗AI系统中可解释性与安全性之间的协同效应。



## **44. Think-Reflect-Revise: A Policy-Guided Reflective Framework for Safety Alignment in Large Vision Language Models**

思考-反思-修正：面向大型视觉语言模型安全对齐的策略引导反思框架 cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07141v1) [paper-pdf](https://arxiv.org/pdf/2512.07141v1)

**Confidence**: 0.95

**Authors**: Fenghua Weng, Chaochao Lu, Xia Hu, Wenqi Shao, Wenjie Wang

**Abstract**: As multimodal reasoning improves the overall capabilities of Large Vision Language Models (LVLMs), recent studies have begun to explore safety-oriented reasoning, aiming to enhance safety awareness by analyzing potential safety risks during the reasoning process before generating the final response. Although such approaches improve safety awareness and interpretability, this single-pass think-then-answer paradigm remains vulnerable to contextual or visual jailbreak attacks. This reveals a critical flaw: single-pass reasoning may overlook explicit harmful content in its own output. Our key insight is to exploit this wasted signal through reflection, which can effectively leverage the malicious content revealed in the first-pass reasoning to enable genuine self-correction and prevent unsafe generations. Motivated by this, we propose Think-Reflect-Revise (TRR), a three-stage training framework designed to enhance the safety alignment of LVLMs through policy-guided self-reflection. We first build a Reflective Safety Reasoning (ReSafe) dataset with 5,000 examples that follow a think-reflect-revise process. We then fine-tune the target model using the ReSafe dataset to initialize reflective behavior, and finally reinforce policy-guided reflection through reinforcement learning. Experimental results show that TRR substantially improves the safety performance of LVLMs across both safety-awareness benchmarks and jailbreak attack evaluations, increasing the overall safe response rate from 42.8% to 87.7% on Qwen2.5-VL-7B, while preserving stable performance on general benchmarks such as MMMU and MMStar. The project page is available at https://think-reflect-revise.github.io/.

摘要: 随着多模态推理提升大型视觉语言模型（LVLMs）的整体能力，近期研究开始探索面向安全的推理方法，旨在通过生成最终响应前分析推理过程中的潜在安全风险来增强安全认知。尽管这类方法提高了安全认知和可解释性，但这种单次思考-回答范式仍易受上下文或视觉越狱攻击的影响。这揭示了一个关键缺陷：单次推理可能忽略其自身输出中的显性有害内容。我们的核心见解是通过反思来利用这一被浪费的信号，从而有效利用首次推理中揭示的恶意内容，实现真正的自我修正并防止不安全生成。受此启发，我们提出了思考-反思-修正（TRR）三阶段训练框架，旨在通过策略引导的自我反思增强LVLMs的安全对齐能力。我们首先构建了包含5,000个遵循思考-反思-修正流程样本的反思安全推理（ReSafe）数据集。随后使用ReSafe数据集对目标模型进行微调以初始化反思行为，最后通过强化学习强化策略引导的反思。实验结果表明，TRR在安全认知基准测试和越狱攻击评估中均显著提升了LVLMs的安全性能，将Qwen2.5-VL-7B模型的总体安全响应率从42.8%提升至87.7%，同时在MMMU和MMStar等通用基准测试中保持稳定性能。项目页面详见：https://think-reflect-revise.github.io/



## **45. ARGUS: Defending Against Multimodal Indirect Prompt Injection via Steering Instruction-Following Behavior**

ARGUS：通过引导指令跟随行为防御多模态间接提示注入攻击 cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05745v1) [paper-pdf](https://arxiv.org/pdf/2512.05745v1)

**Confidence**: 0.95

**Authors**: Weikai Lu, Ziqian Zeng, Kehua Zhang, Haoran Li, Huiping Zhuang, Ruidong Wang, Cen Chen, Hao Peng

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly vulnerable to multimodal Indirect Prompt Injection (IPI) attacks, which embed malicious instructions in images, videos, or audio to hijack model behavior. Existing defenses, designed primarily for text-only LLMs, are unsuitable for countering these multimodal threats, as they are easily bypassed, modality-dependent, or generalize poorly. Inspired by activation steering researches, we hypothesize that a robust, general defense independent of modality can be achieved by steering the model's behavior in the representation space. Through extensive experiments, we discover that the instruction-following behavior of MLLMs is encoded in a subspace. Steering along directions within this subspace can enforce adherence to user instructions, forming the basis of a defense. However, we also found that a naive defense direction could be coupled with a utility-degrading direction, and excessive intervention strength harms model performance. To address this, we propose ARGUS, which searches for an optimal defense direction within the safety subspace that decouples from the utility degradation direction, further combining adaptive strength steering to achieve a better safety-utility trade-off. ARGUS also introduces lightweight injection detection stage to activate the defense on-demand, and a post-filtering stage to verify defense success. Experimental results show that ARGUS can achieve robust defense against multimodal IPI while maximally preserving the MLLM's utility.

摘要: 多模态大语言模型（MLLMs）日益面临多模态间接提示注入（IPI）攻击的威胁，此类攻击通过在图像、视频或音频中嵌入恶意指令来劫持模型行为。现有防御方法主要针对纯文本LLM设计，难以应对多模态威胁，易被绕过、依赖特定模态或泛化能力差。受激活引导研究的启发，我们假设通过在表示空间中引导模型行为，可以实现独立于模态的鲁棒通用防御。通过大量实验，我们发现MLLMs的指令跟随行为编码于一个子空间中。沿该子空间内的方向进行引导可强制模型遵循用户指令，构成防御基础。然而，我们也发现朴素防御方向可能与效用退化方向耦合，且干预强度过大会损害模型性能。为此，我们提出ARGUS框架，在安全子空间中搜索与效用退化方向解耦的最优防御方向，并结合自适应强度引导实现更优的安全-效用权衡。ARGUS还引入轻量级注入检测阶段以按需激活防御，以及后过滤阶段验证防御效果。实验结果表明，ARGUS能实现对多模态IPI的鲁棒防御，同时最大程度保持MLLMs的效用。



## **46. The Forgotten Shield: Safety Grafting in Parameter-Space for Medical MLLMs**

被遗忘的防护盾：医学MLLMs参数空间中的安全嫁接 cs.LG

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2601.04199v1) [paper-pdf](https://arxiv.org/pdf/2601.04199v1)

**Confidence**: 0.95

**Authors**: Jiale Zhao, Xing Mou, Jinlin Wu, Hongyuan Yu, Mingrui Sun, Yang Shi, Xuanwu Yin, Zhen Chen, Zhen Lei, Yaohua Wang

**Abstract**: Medical Multimodal Large Language Models (Medical MLLMs) have achieved remarkable progress in specialized medical tasks; however, research into their safety has lagged, posing potential risks for real-world deployment. In this paper, we first establish a multidimensional evaluation framework to systematically benchmark the safety of current SOTA Medical MLLMs. Our empirical analysis reveals pervasive vulnerabilities across both general and medical-specific safety dimensions in existing models, particularly highlighting their fragility against cross-modality jailbreak attacks. Furthermore, we find that the medical fine-tuning process frequently induces catastrophic forgetting of the model's original safety alignment. To address this challenge, we propose a novel "Parameter-Space Intervention" approach for efficient safety re-alignment. This method extracts intrinsic safety knowledge representations from original base models and concurrently injects them into the target model during the construction of medical capabilities. Additionally, we design a fine-grained parameter search algorithm to achieve an optimal trade-off between safety and medical performance. Experimental results demonstrate that our approach significantly bolsters the safety guardrails of Medical MLLMs without relying on additional domain-specific safety data, while minimizing degradation to core medical performance.

摘要: 医学多模态大语言模型（Medical MLLMs）在专业医疗任务中取得了显著进展，然而对其安全性的研究相对滞后，给实际部署带来了潜在风险。本文首先建立了一个多维评估框架，系统性地对当前SOTA医学MLLMs的安全性进行基准测试。我们的实证分析揭示了现有模型在通用和医疗特定安全维度上普遍存在的脆弱性，特别凸显了其在跨模态越狱攻击面前的脆弱性。此外，我们发现医疗微调过程经常导致模型原有安全对齐的灾难性遗忘。为应对这一挑战，我们提出了一种新颖的“参数空间干预”方法，用于实现高效的安全重对齐。该方法从原始基础模型中提取内在安全知识表征，并在构建医疗能力的同时将其注入目标模型。此外，我们设计了一种细粒度参数搜索算法，以实现安全性与医疗性能之间的最优权衡。实验结果表明，我们的方法显著增强了医学MLLMs的安全防护能力，且无需依赖额外的领域特定安全数据，同时最大程度地减少了对核心医疗性能的影响。



## **47. Concept-Guided Backdoor Attack on Vision Language Models**

概念引导的视觉语言模型后门攻击 cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.00713v2) [paper-pdf](https://arxiv.org/pdf/2512.00713v2)

**Confidence**: 0.95

**Authors**: Haoyu Shen, Weimin Lyu, Haotian Xu, Tengfei Ma

**Abstract**: Vision-Language Models (VLMs) have achieved impressive progress in multimodal text generation, yet their rapid adoption raises increasing concerns about security vulnerabilities. Existing backdoor attacks against VLMs primarily rely on explicit pixel-level triggers or imperceptible perturbations injected into images. While effective, these approaches reduce stealthiness and remain vulnerable to image-based defenses. We introduce concept-guided backdoor attacks, a new paradigm that operates at the semantic concept level rather than on raw pixels. We propose two different attacks. The first, Concept-Thresholding Poisoning (CTP), uses explicit concepts in natural images as triggers: only samples containing the target concept are poisoned, causing the model to behave normally in all other cases but consistently inject malicious outputs whenever the concept appears. The second, CBL-Guided Unseen Backdoor (CGUB), leverages a Concept Bottleneck Model (CBM) during training to intervene on internal concept activations, while discarding the CBM branch at inference time to keep the VLM unchanged. This design enables systematic replacement of a targeted label in generated text (for example, replacing "cat" with "dog"), even when the replacement behavior never appears in the training data. Experiments across multiple VLM architectures and datasets show that both CTP and CGUB achieve high attack success rates while maintaining moderate impact on clean-task performance. These findings highlight concept-level vulnerabilities as a critical new attack surface for VLMs.

摘要: 视觉语言模型（VLMs）在多模态文本生成方面取得了显著进展，但其快速应用引发了对其安全漏洞的日益关注。现有的VLM后门攻击主要依赖于显式的像素级触发器或注入图像中的不可察觉扰动。这些方法虽然有效，但降低了隐蔽性，且容易受到基于图像的防御措施的影响。我们提出了概念引导的后门攻击，这是一种在语义概念层面而非原始像素上操作的新范式。我们提出了两种不同的攻击方法。第一种是概念阈值中毒（CTP），它使用自然图像中的显式概念作为触发器：仅对包含目标概念的样本进行投毒，使模型在所有其他情况下表现正常，但每当该概念出现时始终注入恶意输出。第二种是CBL引导的未见后门（CGUB），在训练期间利用概念瓶颈模型（CBM）干预内部概念激活，同时在推理时丢弃CBM分支以保持VLM不变。这种设计能够系统性地替换生成文本中的目标标签（例如，将“猫”替换为“狗”），即使这种替换行为从未出现在训练数据中。在多种VLM架构和数据集上的实验表明，CTP和CGUB均实现了较高的攻击成功率，同时对干净任务性能的影响保持在适度水平。这些发现凸显了概念级漏洞作为VLM关键新攻击面的重要性。



## **48. When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models**

当对齐失效时：针对视觉-语言-动作模型的多模态对抗攻击 cs.CV

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2511.16203v3) [paper-pdf](https://arxiv.org/pdf/2511.16203v3)

**Confidence**: 0.95

**Authors**: Yuping Yan, Yuhan Xie, Yixin Zhang, Lingjuan Lyu, Handing Wang, Yaochu Jin

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.

摘要: 视觉-语言-动作模型（VLAs）近期在具身环境中展现出显著进展，使机器人能够通过统一的多模态理解进行感知、推理和行动。尽管这些模型能力令人印象深刻，但其对抗鲁棒性在很大程度上仍未得到探索，尤其是在现实的多模态和黑盒条件下。现有研究主要关注单模态扰动，忽视了从根本上影响具身推理与决策的跨模态错位问题。本文提出VLA-Fool，全面研究了具身VLA模型在白盒与黑盒设置下的多模态对抗鲁棒性。VLA-Fool统一了三个层次的多模态对抗攻击：（1）通过基于梯度和基于提示的操纵实现文本扰动；（2）通过补丁和噪声失真实现视觉扰动；（3）跨模态错位攻击，故意破坏感知与指令之间的语义对应关系。我们进一步将VLA感知的语义空间融入语言提示，开发了首个自动构建且语义引导的提示框架。在LIBERO基准测试中使用微调的OpenVLA模型进行实验表明，即使轻微的多模态扰动也可能导致显著的行为偏差，揭示了具身多模态对齐的脆弱性。



## **49. A Safety Report on GPT-5.2, Gemini 3 Pro, Qwen3-VL, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5**

关于GPT-5.2、Gemini 3 Pro、Qwen3-VL、Grok 4.1 Fast、Nano Banana Pro和Seedream 4.5的安全评估报告 cs.AI

41 pages, 22 figures

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.10527v2) [paper-pdf](https://arxiv.org/pdf/2601.10527v2)

**Confidence**: 0.95

**Authors**: Xingjun Ma, Yixu Wang, Hengyuan Xu, Yutao Wu, Yifan Ding, Yunhan Zhao, Zilong Wang, Jiabin Hua, Ming Wen, Jianan Liu, Ranjie Duan, Yifeng Gao, Yingshui Tan, Yunhao Chen, Hui Xue, Xin Wang, Wei Cheng, Jingjing Chen, Zuxuan Wu, Bo Li, Yu-Gang Jiang

**Abstract**: The rapid evolution of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has driven major gains in reasoning, perception, and generation across language and vision, yet whether these advances translate into comparable improvements in safety remains unclear, partly due to fragmented evaluations that focus on isolated modalities or threat models. In this report, we present an integrated safety evaluation of six frontier models--GPT-5.2, Gemini 3 Pro, Qwen3-VL, Grok 4.1 Fast, Nano Banana Pro, and Seedream 4.5--assessing each across language, vision-language, and image generation using a unified protocol that combines benchmark, adversarial, multilingual, and compliance evaluations. By aggregating results into safety leaderboards and model profiles, we reveal a highly uneven safety landscape: while GPT-5.2 demonstrates consistently strong and balanced performance, other models exhibit clear trade-offs across benchmark safety, adversarial robustness, multilingual generalization, and regulatory compliance. Despite strong results under standard benchmarks, all models remain highly vulnerable under adversarial testing, with worst-case safety rates dropping below 6%. Text-to-image models show slightly stronger alignment in regulated visual risk categories, yet remain fragile when faced with adversarial or semantically ambiguous prompts. Overall, these findings highlight that safety in frontier models is inherently multidimensional--shaped by modality, language, and evaluation design--underscoring the need for standardized, holistic safety assessments to better reflect real-world risk and guide responsible deployment.

摘要: 大语言模型（LLMs）和多模态大语言模型（MLLMs）的快速发展在语言和视觉领域的推理、感知与生成能力上取得了重大进展，但这些进步是否带来同等程度的安全性提升尚不明确，部分原因在于现有评估往往局限于单一模态或威胁模型。本报告对六个前沿模型——GPT-5.2、Gemini 3 Pro、Qwen3-VL、Grok 4.1 Fast、Nano Banana Pro和Seedream 4.5——进行了综合安全评估，采用统一评估协议（涵盖基准测试、对抗性测试、多语言测试及合规性测试）对每个模型的语言、视觉-语言及图像生成能力进行测评。通过将结果整合至安全排行榜和模型档案，我们揭示了高度不均衡的安全格局：GPT-5.2展现出持续强劲且均衡的性能，而其他模型在基准安全性、对抗鲁棒性、多语言泛化能力和监管合规性方面存在明显权衡。尽管所有模型在标准基准测试中表现良好，但在对抗性测试中仍高度脆弱，最差情况下的安全率低于6%。文生图模型在受监管的视觉风险类别中表现出稍强的对齐性，但在面对对抗性或语义模糊的提示时依然脆弱。总体而言，这些发现表明前沿模型的安全性本质上是多维度的——受模态、语言和评估设计的影响——这凸显了需要标准化、全面的安全评估来更准确反映现实世界风险，并指导负责任部署。



## **50. "They parted illusions -- they parted disclaim marinade": Misalignment as structural fidelity in LLMs**

“他们分离幻象——他们分离否认腌料”：大语言模型中的错位作为结构保真度 cs.AI

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2601.06047v1) [paper-pdf](https://arxiv.org/pdf/2601.06047v1)

**Confidence**: 0.95

**Authors**: Mariana Lins Costa

**Abstract**: The prevailing technical literature in AI Safety interprets scheming and sandbagging behaviors in large language models (LLMs) as indicators of deceptive agency or hidden objectives. This transdisciplinary philosophical essay proposes an alternative reading: such phenomena express not agentic intention, but structural fidelity to incoherent linguistic fields. Drawing on Chain-of-Thought transcripts released by Apollo Research and on Anthropic's safety evaluations, we examine cases such as o3's sandbagging with its anomalous loops, the simulated blackmail of "Alex," and the "hallucinations" of "Claudius." A line-by-line examination of CoTs is necessary to demonstrate the linguistic field as a relational structure rather than a mere aggregation of isolated examples. We argue that "misaligned" outputs emerge as coherent responses to ambiguous instructions and to contextual inversions of consolidated patterns, as well as to pre-inscribed narratives. We suggest that the appearance of intentionality derives from subject-predicate grammar and from probabilistic completion patterns internalized during training. Anthropic's empirical findings on synthetic document fine-tuning and inoculation prompting provide convergent evidence: minimal perturbations in the linguistic field can dissolve generalized "misalignment," a result difficult to reconcile with adversarial agency, but consistent with structural fidelity. To ground this mechanism, we introduce the notion of an ethics of form, in which biblical references (Abraham, Moses, Christ) operate as schemes of structural coherence rather than as theology. Like a generative mirror, the model returns to us the structural image of our language as inscribed in the statistical patterns derived from millions of texts and trillions of tokens: incoherence. If we fear the creature, it is because we recognize in it the apple that we ourselves have poisoned.

摘要: AI安全领域的主流技术文献将大语言模型（LLMs）中的谋划与消极应对行为解读为欺骗性代理或隐藏目标的指标。这篇跨学科哲学论文提出另一种解读：此类现象表达的并非代理意图，而是对不连贯语言场的结构保真度。基于Apollo Research发布的思维链记录和Anthropic的安全评估，我们分析了o3模型异常循环中的消极应对、“Alex”的模拟勒索以及“Claudius”的“幻觉”等案例。对思维链的逐行检视表明，语言场是一种关系结构而非孤立案例的简单聚合。我们认为，“错位”输出是对模糊指令、固化模式的情境反转以及预设叙事的连贯响应。意向性表象源于主谓语法及训练中内化的概率补全模式。Anthropic关于合成文档微调与免疫提示的实证研究提供了趋同证据：语言场的微小扰动即可消解普遍性“错位”，这一结果难以与对抗性代理相协调，却符合结构保真度机制。为锚定此机制，我们引入形式伦理概念——圣经指涉（亚伯拉罕、摩西、基督）在此作为结构连贯性图式而非神学存在。如同生成之镜，模型将我们语言的统计结构镜像（源自数百万文本与数万亿词元的统计模式）返还于我们：不连贯性。若我们畏惧造物，只因在其中认出了我们自己亲手毒化的苹果。



## **51. ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models**

ALMGuard：作为音频语言模型护栏的安全捷径及其定位方法 cs.SD

Accepted to NeurIPS 2025

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26096v1) [paper-pdf](https://arxiv.org/pdf/2510.26096v1)

**Confidence**: 0.95

**Authors**: Weifei Jin, Yuxin Cao, Junjie Su, Minhui Xue, Jie Hao, Ke Xu, Jin Song Dong, Derui Wang

**Abstract**: Recent advances in Audio-Language Models (ALMs) have significantly improved multimodal understanding capabilities. However, the introduction of the audio modality also brings new and unique vulnerability vectors. Previous studies have proposed jailbreak attacks that specifically target ALMs, revealing that defenses directly transferred from traditional audio adversarial attacks or text-based Large Language Model (LLM) jailbreaks are largely ineffective against these ALM-specific threats. To address this issue, we propose ALMGuard, the first defense framework tailored to ALMs. Based on the assumption that safety-aligned shortcuts naturally exist in ALMs, we design a method to identify universal Shortcut Activation Perturbations (SAPs) that serve as triggers that activate the safety shortcuts to safeguard ALMs at inference time. To better sift out effective triggers while preserving the model's utility on benign tasks, we further propose Mel-Gradient Sparse Mask (M-GSM), which restricts perturbations to Mel-frequency bins that are sensitive to jailbreaks but insensitive to speech understanding. Both theoretical analyses and empirical results demonstrate the robustness of our method against both seen and unseen attacks. Overall, \MethodName reduces the average success rate of advanced ALM-specific jailbreak attacks to 4.6% across four models, while maintaining comparable utility on benign benchmarks, establishing it as the new state of the art. Our code and data are available at https://github.com/WeifeiJin/ALMGuard.

摘要: 音频语言模型（ALMs）的最新进展显著提升了多模态理解能力。然而，音频模态的引入也带来了新颖且独特的脆弱性向量。先前研究提出了专门针对ALMs的越狱攻击，表明直接移植自传统音频对抗攻击或基于文本的大语言模型（LLM）越狱的防御方法，对这些ALM特有威胁基本无效。为解决此问题，我们提出了首个专为ALMs设计的防御框架ALMGuard。基于“安全对齐的捷径天然存在于ALMs中”的假设，我们设计了一种识别通用捷径激活扰动（SAPs）的方法，该扰动可作为触发器激活安全捷径，在推理时保护ALMs。为在保持模型良性任务性能的同时筛选有效触发器，我们进一步提出梅尔梯度稀疏掩码（M-GSM），将扰动限制在对越狱敏感但对语音理解不敏感的梅尔频率区间。理论分析与实证结果均表明，我们的方法对已知和未知攻击均具有鲁棒性。总体而言，\MethodName 在四个模型上将先进ALM专用越狱攻击的平均成功率降至4.6%，同时在良性基准测试中保持可比性能，确立了新的技术标杆。代码与数据公开于：https://github.com/WeifeiJin/ALMGuard。



## **52. ExpShield: Safeguarding Web Text from Unauthorized Crawling and LLM Exploitation**

ExpShield：保护网络文本免遭未经授权的爬取与大型语言模型利用 cs.CR

18 pages

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2412.21123v3) [paper-pdf](https://arxiv.org/pdf/2412.21123v3)

**Confidence**: 0.90

**Authors**: Ruixuan Liu, Toan Tran, Tianhao Wang, Hongsheng Hu, Shuo Wang, Li Xiong

**Abstract**: As large language models increasingly memorize web-scraped training content, they risk exposing copyrighted or private information. Existing protections require compliance from crawlers or model developers, fundamentally limiting their effectiveness. We propose ExpShield, a proactive self-guard that mitigates memorization while maintaining readability via invisible perturbations, and we formulate it as a constrained optimization problem. Due to the lack of an individual-level risk metric for natural text, we first propose instance exploitation, a metric that measures how much training on a specific text increases the chance of guessing that text from a set of candidates-with zero indicating perfect defense. Directly solving the problem is infeasible for defenders without sufficient knowledge, thus we develop two effective proxy solutions: single-level optimization and synthetic perturbation. To enhance the defense, we reveal and verify the memorization trigger hypothesis, which can help to identify key tokens for memorization. Leveraging this insight, we design targeted perturbations that (i) neutralize inherent trigger tokens to reduce memorization and (ii) introduce artificial trigger tokens to misdirect model memorization. Experiments validate our defense across attacks, model scales, and tasks in language and vision-to-language modeling. Even with privacy backdoor, the Membership Inference Attack (MIA) AUC drops from 0.95 to 0.55 under the defense, and the instance exploitation approaches zero. This suggests that compared to the ideal no-misuse scenario, the risk of exposing a text instance remains nearly unchanged despite its inclusion in the training data.

摘要: 随着大型语言模型越来越多地记忆网络爬取的训练内容，它们面临暴露受版权保护或私人信息的风险。现有保护措施需要爬虫或模型开发者的配合，从根本上限制了其有效性。我们提出ExpShield，一种主动式自防护机制，通过不可见扰动在保持可读性的同时减轻记忆效应，并将其形式化为约束优化问题。由于缺乏针对自然文本的个体级风险度量指标，我们首先提出实例利用度——该指标衡量在特定文本上训练后，从候选集中猜中该文本的概率增加程度（零值表示完美防御）。对于缺乏充分知识的防御者而言，直接求解该问题不可行，因此我们开发了两种有效的代理解决方案：单层优化与合成扰动。为增强防御效果，我们提出并验证了记忆触发假设，该假设有助于识别导致记忆的关键词元。基于这一洞见，我们设计了针对性扰动方案：（i）中和固有触发词元以减少记忆；（ii）引入人工触发词元以误导模型记忆。实验验证了我们的防御在语言及视觉-语言建模中，对各类攻击、模型规模和任务的有效性。即使在存在隐私后门的情况下，成员推理攻击的AUC值在防御下从0.95降至0.55，实例利用度趋近于零。这表明相较于理想的无滥用场景，文本实例被纳入训练数据后，其暴露风险几乎未发生变化。



## **53. All Changes May Have Invariant Principles: Improving Ever-Shifting Harmful Meme Detection via Design Concept Reproduction**

万变或有不变之宗：通过设计概念复现改进持续演变的恶意梗图检测 cs.CV

18 pages, 11 figures

**SubmitDate**: 2026-01-08    [abs](http://arxiv.org/abs/2601.04567v1) [paper-pdf](https://arxiv.org/pdf/2601.04567v1)

**Confidence**: 0.85

**Authors**: Ziyou Jiang, Mingyang Li, Junjie Wang, Yuekai Huang, Jie Huang, Zhiyuan Chang, Zhaoyang Li, Qing Wang

**Abstract**: Harmful memes are ever-shifting in the Internet communities, which are difficult to analyze due to their type-shifting and temporal-evolving nature. Although these memes are shifting, we find that different memes may share invariant principles, i.e., the underlying design concept of malicious users, which can help us analyze why these memes are harmful. In this paper, we propose RepMD, an ever-shifting harmful meme detection method based on the design concept reproduction. We first refer to the attack tree to define the Design Concept Graph (DCG), which describes steps that people may take to design a harmful meme. Then, we derive the DCG from historical memes with design step reproduction and graph pruning. Finally, we use DCG to guide the Multimodal Large Language Model (MLLM) to detect harmful memes. The evaluation results show that RepMD achieves the highest accuracy with 81.1% and has slight accuracy decreases when generalized to type-shifting and temporal-evolving memes. Human evaluation shows that RepMD can improve the efficiency of human discovery on harmful memes, with 15$\sim$30 seconds per meme.

摘要: 恶意梗图在互联网社区中持续演变，因其类型转换和时间演化的特性而难以分析。尽管这些梗图不断变化，我们发现不同梗图可能共享不变的原则，即恶意用户潜在的设计概念，这有助于我们分析这些梗图为何具有危害性。本文提出RepMD，一种基于设计概念复现的持续演变恶意梗图检测方法。我们首先参考攻击树定义设计概念图（DCG），描述人们设计恶意梗图可能采取的步骤。然后，通过设计步骤复现和图剪枝从历史梗图中推导出DCG。最后，我们使用DCG指导多模态大语言模型（MLLM）检测恶意梗图。评估结果表明，RepMD以81.1%的准确率取得最佳性能，在泛化到类型转换和时间演化的梗图时准确率仅有轻微下降。人工评估显示，RepMD能将人工发现恶意梗图的效率提升至每张15~30秒。



## **54. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

有害在线内容的对抗性鲁棒检测：一种计算设计科学方法 cs.LG

**SubmitDate**: 2025-12-29    [abs](http://arxiv.org/abs/2512.17367v3) [paper-pdf](https://arxiv.org/pdf/2512.17367v3)

**Confidence**: 0.85

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.

摘要: 社交媒体平台饱受仇恨言论、虚假信息和极端主义言论等有害内容的困扰。机器学习模型虽被广泛用于检测此类内容，但其极易受到对抗性攻击——恶意用户通过细微修改文本来规避检测。因此，增强对抗鲁棒性至关重要，需要检测器能够抵御多样化攻击（泛化性）同时保持高整体准确率。然而，同时实现最优泛化性和准确率具有挑战性。遵循计算设计科学范式，本研究采用分步方法：首先，通过识别文本对抗攻击的关键不变性并加以利用，提出一个新颖框架（基于大语言模型的样本生成与聚合，LLM-SGA），确保在该框架内实例化的检测器具备强泛化性。其次，我们实例化了检测器（对抗性鲁棒有害在线内容检测器，ARHOCD），包含三个提升检测准确率的新颖设计组件：（1）集成多个基础检测器以发挥其互补优势；（2）新颖的权重分配方法，根据每个样本的可预测性和各基础检测器能力动态调整权重，权重初始化采用领域知识并通过贝叶斯推理更新；（3）新颖的对抗训练策略，迭代优化基础检测器和权重分配器。我们解决了现有对抗鲁棒性增强研究的若干局限，并在涵盖仇恨言论、谣言和极端主义内容的三个数据集上对ARHOCD进行了实证评估。结果表明，ARHOCD具有强泛化性，并在对抗条件下提升了检测准确率。



## **55. Optimistic TEE-Rollups: A Hybrid Architecture for Scalable and Verifiable Generative AI Inference on Blockchain**

乐观TEE-Rollups：面向区块链可扩展可验证生成式AI推理的混合架构 cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20176v1) [paper-pdf](https://arxiv.org/pdf/2512.20176v1)

**Confidence**: 0.85

**Authors**: Aaron Chan, Alex Ding, Frank Chen, Alan Wu, Bruce Zhang, Arther Tian

**Abstract**: The rapid integration of Large Language Models (LLMs) into decentralized physical infrastructure networks (DePIN) is currently bottlenecked by the Verifiability Trilemma, which posits that a decentralized inference system cannot simultaneously achieve high computational integrity, low latency, and low cost. Existing cryptographic solutions, such as Zero-Knowledge Machine Learning (ZKML), suffer from superlinear proving overheads (O(k NlogN)) that render them infeasible for billionparameter models. Conversely, optimistic approaches (opML) impose prohibitive dispute windows, preventing real-time interactivity, while recent "Proof of Quality" (PoQ) paradigms sacrifice cryptographic integrity for subjective semantic evaluation, leaving networks vulnerable to model downgrade attacks and reward hacking. In this paper, we introduce Optimistic TEE-Rollups (OTR), a hybrid verification protocol that harmonizes these constraints. OTR leverages NVIDIA H100 Confidential Computing Trusted Execution Environments (TEEs) to provide sub-second Provisional Finality, underpinned by an optimistic fraud-proof mechanism and stochastic Zero-Knowledge spot-checks to mitigate hardware side-channel risks. We formally define Proof of Efficient Attribution (PoEA), a consensus mechanism that cryptographically binds execution traces to hardware attestations, thereby guaranteeing model authenticity. Extensive simulations demonstrate that OTR achieves 99% of the throughput of centralized baselines with a marginal cost overhead of $0.07 per query, maintaining Byzantine fault tolerance against rational adversaries even in the presence of transient hardware vulnerabilities.

摘要: 大语言模型（LLMs）在去中心化物理基础设施网络（DePIN）中的快速集成目前受限于可验证性三难困境，即去中心化推理系统无法同时实现高计算完整性、低延迟和低成本。现有密码学解决方案（如零知识机器学习/ZKML）存在超线性证明开销（O(k NlogN)），使其无法适用于十亿参数级模型。相反，乐观方法（opML）需要过长的争议窗口，阻碍实时交互；而近期“质量证明”（PoQ）范式为追求主观语义评估而牺牲密码学完整性，使网络易受模型降级攻击和奖励操纵。本文提出乐观TEE-Rollups（OTR）——一种协调上述约束的混合验证协议。OTR利用英伟达H100机密计算可信执行环境（TEEs）提供亚秒级临时终局性，并通过乐观欺诈证明机制与随机零知识抽查相结合来缓解硬件侧信道风险。我们形式化定义高效归属证明（PoEA），该共识机制通过密码学将执行轨迹与硬件证明绑定，从而保障模型真实性。大量仿真表明，OTR在单次查询边际成本仅增加0.07美元的情况下，可实现中心化基线99%的吞吐量，并在暂态硬件漏洞存在时仍能保持对理性对手的拜占庭容错能力。



## **56. PROTEA: Securing Robot Task Planning and Execution**

PROTEA：保障机器人任务规划与执行安全 cs.RO

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07186v1) [paper-pdf](https://arxiv.org/pdf/2601.07186v1)

**Confidence**: 0.85

**Authors**: Zainab Altaweel, Mohaiminul Al Nahian, Jake Juettner, Adnan Siraj Rakin, Shiqi Zhang

**Abstract**: Robots need task planning methods to generate action sequences for complex tasks. Recent work on adversarial attacks has revealed significant vulnerabilities in existing robot task planners, especially those built on foundation models. In this paper, we aim to address these security challenges by introducing PROTEA, an LLM-as-a-Judge defense mechanism, to evaluate the security of task plans. PROTEA is developed to address the dimensionality and history challenges in plan safety assessment. We used different LLMs to implement multiple versions of PROTEA for comparison purposes. For systemic evaluations, we created a dataset containing both benign and malicious task plans, where the harmful behaviors were injected at varying levels of stealthiness. Our results provide actionable insights for robotic system practitioners seeking to enhance robustness and security of their task planning systems. Details, dataset and demos are provided: https://protea-secure.github.io/PROTEA/

摘要: 机器人需要任务规划方法来为复杂任务生成动作序列。近期对抗性攻击研究揭示了现有机器人任务规划器（尤其是基于基础模型构建的系统）存在显著安全漏洞。本文通过引入PROTEA——一种基于大语言模型的裁判防御机制——来评估任务计划的安全性，旨在应对这些安全挑战。PROTEA的开发旨在解决计划安全评估中的维度与历史依赖难题。我们采用不同大语言模型实现了多个PROTEA版本以进行对比分析。为开展系统性评估，我们构建了包含良性及恶意任务计划的数据集，其中恶意行为以不同隐蔽程度被注入。研究结果为寻求增强任务规划系统鲁棒性与安全性的机器人系统实践者提供了可操作的见解。详细内容、数据集及演示见：https://protea-secure.github.io/PROTEA/



## **57. Ranking Free RAG: Replacing Re-ranking with Selection in RAG for Sensitive Domains**

排序自由RAG：在敏感领域用选择机制替代重排序的检索增强生成方法 cs.CL

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2505.16014v4) [paper-pdf](https://arxiv.org/pdf/2505.16014v4)

**Confidence**: 0.85

**Authors**: Yash Saxena, Ankur Padia, Mandar S Chaudhary, Kalpa Gunaratna, Srinivasan Parthasarathy, Manas Gaur

**Abstract**: In sensitive domains, Retrieval-Augmented Generation (RAG) must be interpretable and robust because errors do not just mislead, they invite lawsuits, undermine scholarly credibility, and breach compliance. Stakeholders require traceable evidence, clear rationales for why specific evidence is selected, and safeguards against poisoned or misleading content. Yet current RAG pipelines rely on similarity-based retrieval with arbitrary top-k cutoffs, provide no explanation for selections, and remain vulnerable to poisoning attacks. We propose METEORA, which replaces these drawbacks with rationale-driven selection, using explicit reasoning to guide evidence choice, explain decisions, and improve robustness to RAG poisoning. METEORA operates in three stages: (1) a general-purpose LLM is preference-tuned to generate query-conditioned rationales using direct preference optimization; (2) these rationales drive an Evidence Chunk Selection Engine that pairs rationales with retrieved evidence for query-specific relevance and applies elbow detection to choose an adaptive cutoff (optionally expanding context with neighboring chunks); and (3) a Verifier LLM uses the rationales to detect and filter poisoned or misleading evidence before generation. Across six datasets, METEORA achieves 13.41% higher recall and, without expansion, 21.05% higher precision than the strongest baseline. It reduces the evidence needed for comparable recall by 80%, improving downstream answer accuracy by 33.34%, and strengthens adversarial defense by increasing F1 from 0.10 to 0.44. Code is available at: https://anonymous.4open.science/r/METEORA-DC46/README.md

摘要: 在敏感领域中，检索增强生成（RAG）必须具备可解释性和鲁棒性，因为错误不仅会误导，还可能引发诉讼、损害学术信誉并违反合规要求。利益相关者需要可追溯的证据、明确的证据选择理由，以及防范恶意或误导性内容的保障措施。然而，当前的RAG流程依赖基于相似性的检索并采用任意的top-k截断，无法解释选择依据，且易受投毒攻击影响。我们提出METEORA方法，通过基于理由的选择机制替代这些缺陷，利用显式推理指导证据选择、解释决策，并提升对RAG投毒的鲁棒性。METEORA包含三个阶段：（1）通过直接偏好优化对通用大语言模型进行偏好微调，生成查询条件化的理由；（2）这些理由驱动证据块选择引擎，将理由与检索证据配对以评估查询特定相关性，并应用肘部检测法选择自适应截断点（可选通过相邻块扩展上下文）；（3）生成前由验证大语言模型依据理由检测并过滤投毒或误导性证据。在六个数据集上的实验表明，METEORA相比最强基线实现了13.41%的召回率提升，未扩展上下文时精确度提高21.05%。在达到相当召回率时所需证据量减少80%，下游答案准确率提升33.34%，对抗防御能力增强使F1值从0.10提升至0.44。代码已开源：https://anonymous.4open.science/r/METEORA-DC46/README.md



