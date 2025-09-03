# Latest Large Language Model Attack Papers
**update at 2025-09-03 10:19:57**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. See No Evil: Adversarial Attacks Against Linguistic-Visual Association in Referring Multi-Object Tracking Systems**

看不到邪恶：引用多对象跟踪系统中对语言视觉关联的对抗攻击 cs.CV

12 pages, 1 figure, 3 tables

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02028v1) [paper-pdf](http://arxiv.org/pdf/2509.02028v1)

**Authors**: Halima Bouzidi, Haoyu Liu, Mohammad Al Faruque

**Abstract**: Language-vision understanding has driven the development of advanced perception systems, most notably the emerging paradigm of Referring Multi-Object Tracking (RMOT). By leveraging natural-language queries, RMOT systems can selectively track objects that satisfy a given semantic description, guided through Transformer-based spatial-temporal reasoning modules. End-to-End (E2E) RMOT models further unify feature extraction, temporal memory, and spatial reasoning within a Transformer backbone, enabling long-range spatial-temporal modeling over fused textual-visual representations. Despite these advances, the reliability and robustness of RMOT remain underexplored. In this paper, we examine the security implications of RMOT systems from a design-logic perspective, identifying adversarial vulnerabilities that compromise both the linguistic-visual referring and track-object matching components. Additionally, we uncover a novel vulnerability in advanced RMOT models employing FIFO-based memory, whereby targeted and consistent attacks on their spatial-temporal reasoning introduce errors that persist within the history buffer over multiple subsequent frames. We present VEIL, a novel adversarial framework designed to disrupt the unified referring-matching mechanisms of RMOT models. We show that carefully crafted digital and physical perturbations can corrupt the tracking logic reliability, inducing track ID switches and terminations. We conduct comprehensive evaluations using the Refer-KITTI dataset to validate the effectiveness of VEIL and demonstrate the urgent need for security-aware RMOT designs for critical large-scale applications.

摘要: 图像视觉理解推动了高级感知系统的发展，最引人注目的是参考多对象跟踪（RMOT）的新兴范式。通过利用自然语言查询，RMOT系统可以在基于Transformer的时空推理模块的指导下选择性地跟踪满足给定语义描述的对象。端到端（E2 E）RMOT模型进一步统一了Transformer骨干中的特征提取、时间记忆和空间推理，从而在融合的文本-视觉表示上实现了长距离时空建模。尽管有这些进步，RMOT的可靠性和鲁棒性仍然没有得到充分的研究。在本文中，我们从设计逻辑的角度研究了RMOT系统的安全影响，识别了损害语言视觉引用和跟踪对象匹配组件的对抗漏洞。此外，我们还发现了使用基于FIFO的内存的高级RMOT模型中的一个新漏洞，从而对其时空推理进行有针对性且一致的攻击，从而引入了在历史缓冲区中持续存在的错误。我们提出了VEIL，这是一种新型对抗框架，旨在破坏RMOT模型的统一触发匹配机制。我们表明，精心设计的数字和物理扰动可能会破坏跟踪逻辑的可靠性，导致轨道ID开关和端接。我们使用Refer-KITTI数据集进行全面评估，以验证VEIL的有效性，并证明关键大规模应用对安全感知RMOT设计的迫切需求。



## **2. Unraveling LLM Jailbreaks Through Safety Knowledge Neurons**

通过安全知识神经元解开LLM越狱事件 cs.AI

10 pages, 6 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01631v1) [paper-pdf](http://arxiv.org/pdf/2509.01631v1)

**Authors**: Chongwen Zhao, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation, a technique known as "Jailbreak." While some studies have achieved defenses against jailbreak attacks by modifying output distributions or detecting harmful content, the exact rationale still remains elusive. In this work, we present a novel neuron-level interpretability method that focuses on the role of safety-related knowledge neurons. Unlike existing approaches, our method projects the model's internal representation into a more consistent and interpretable vocabulary space. We then show that adjusting the activation of safety-related neurons can effectively control the model's behavior with a mean ASR higher than 97%. Building on this insight, we propose SafeTuning, a fine-tuning strategy that reinforces safety-critical neurons to improve model robustness against jailbreaks. SafeTuning consistently reduces attack success rates across multiple LLMs and outperforms all four baseline defenses. These findings offer a new perspective on understanding and defending against jailbreak attacks.

摘要: 大型语言模型（LLM）在各种应用中越来越受到关注。尽管如此，人们越来越担心，因为一些用户试图利用这些模型进行恶意目的，包括合成受控物质和传播虚假信息，一种被称为“越狱”的技术。“虽然一些研究通过修改输出分发或检测有害内容来实现对越狱攻击的防御，但确切的理由仍然难以捉摸。在这项工作中，我们提出了一种新型的神经元水平解释方法，重点关注与安全相关的知识神经元的作用。与现有方法不同，我们的方法将模型的内部表示投影到更一致和可解释的词汇空间中。然后我们表明，调整安全相关神经元的激活可以有效控制模型的行为，平均ASB高于97%。基于这一见解，我们提出了SafeTuning，这是一种微调策略，可以加强安全关键神经元，以提高模型针对越狱的鲁棒性。SafeTuning持续降低多个LLM的攻击成功率，并优于所有四种基线防御。这些发现为理解和防御越狱攻击提供了新的视角。



## **3. LLMHoney: A Real-Time SSH Honeypot with Large Language Model-Driven Dynamic Response Generation**

LLMHoney：具有大型语言模型驱动动态响应生成的实时SSH蜜罐 cs.CR

7 Pages

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01463v1) [paper-pdf](http://arxiv.org/pdf/2509.01463v1)

**Authors**: Pranjay Malhotra

**Abstract**: Cybersecurity honeypots are deception tools for engaging attackers and gather intelligence, but traditional low or medium-interaction honeypots often rely on static, pre-scripted interactions that can be easily identified by skilled adversaries. This Report presents LLMHoney, an SSH honeypot that leverages Large Language Models (LLMs) to generate realistic, dynamic command outputs in real time. LLMHoney integrates a dictionary-based virtual file system to handle common commands with low latency while using LLMs for novel inputs, achieving a balance between authenticity and performance. We implemented LLMHoney using open-source LLMs and evaluated it on a testbed with 138 representative Linux commands. We report comprehensive metrics including accuracy (exact-match, Cosine Similarity, Jaro-Winkler Similarity, Levenshtein Similarity and BLEU score), response latency and memory overhead. We evaluate LLMHoney using multiple LLM backends ranging from 0.36B to 3.8B parameters, including both open-source models and a proprietary model(Gemini). Our experiments compare 13 different LLM variants; results show that Gemini-2.0 and moderately-sized models Qwen2.5:1.5B and Phi3:3.8B provide the most reliable and accurate responses, with mean latencies around 3 seconds, whereas smaller models often produce incorrect or out-of-character outputs. We also discuss how LLM integration improves honeypot realism and adaptability compared to traditional honeypots, as well as challenges such as occasional hallucinated outputs and increased resource usage. Our findings demonstrate that LLM-driven honeypots are a promising approach to enhance attacker engagement and collect richer threat intelligence.

摘要: 网络安全蜜罐是用于吸引攻击者和收集情报的欺骗工具，但传统的低或中等交互蜜罐通常依赖于静态的、预先脚本化的交互，这些交互可以被熟练的对手轻松识别。本报告介绍了LLMHoney，这是一个SSH蜜罐，利用大型语言模型（LLM）实时生成真实、动态的命令输出。LLMHoney集成了基于字典的虚拟文件系统，以低延迟处理常见命令，同时使用LLM进行新型输入，实现真实性和性能之间的平衡。我们使用开源LLM实现了LLMHoney，并在具有138个代表性的Linux命令的测试床上对其进行了评估。我们报告了全面的指标，包括准确性（精确匹配、Cosine相似性、Jaro-Winkler相似性、Levenshtein相似性和BLEU评分）、响应延迟和内存负载。我们使用从0.36B到3.8B参数的多个LLM后台来评估LLMHoney，包括开源模型和专有模型（Gemini）。我们的实验比较了13种不同的LLM变体;结果表明，Gemini-2.0和中等大小的模型Qwen 2.5：1.5B和Phi 3：3.8B提供了最可靠和准确的响应，平均延迟时间约为3秒，而较小的模型通常会产生错误或不合字符的输出。我们还讨论了与传统蜜罐相比，LLM集成如何提高蜜罐的真实性和适应性，以及偶尔幻觉输出和资源使用增加等挑战。我们的研究结果表明，LLM驱动的蜜罐是增强攻击者参与度和收集更丰富威胁情报的一种有希望的方法。



## **4. MEGen: Generative Backdoor into Large Language Models via Model Editing**

MEGen：通过模型编辑进入大型语言模型的生成后门 cs.CL

ACL 2025 Findings

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2408.10722v2) [paper-pdf](http://arxiv.org/pdf/2408.10722v2)

**Authors**: Jiyang Qiu, Xinbei Ma, Zhuosheng Zhang, Hai Zhao, Yun Li, Qianren Wang

**Abstract**: Large language models (LLMs) have exhibited remarkable versatility and adaptability, while their widespread adoption across various applications also raises critical safety concerns. This paper focuses on the impact of backdoored LLMs. Traditional backdoor injection methods are primarily limited to yes-or-no discriminative tasks, leading users to underestimate the potential risks of backdoored LLMs. Given the inherently generative nature of LLMs, this paper reveals that a generative backdoor injected into LLMs can expose the true safety risks in their applications. We propose an editing-based generative backdoor, named MEGen, aiming to expand the backdoor to generative tasks in a unified format of any text-to any text, leading to natural generations with a specific intention. Experiments show that MEGen achieves a high attack success rate by adjusting only a small set of local parameters with few-shot samples. Notably, we show that the backdoored model, when triggered, can freely output pre-set dangerous information while completing downstream tasks. Our work highlights that MEGen enables backdoors in LLMs to exhibit generative capabilities, causing potential safety risks by altering the generative style. The code is available at https://github.com/MonoQ-hub/MEGen.

摘要: 大型语言模型（LLM）表现出非凡的通用性和适应性，而它们在各种应用程序中的广泛采用也引发了关键的安全问题。本文重点讨论了后门的LLM的影响。传统的后门注入方法主要局限于“是或否”区分性任务，导致用户低估了后门LLM的潜在风险。鉴于LLM固有的生成性，本文揭示了注入LLM的生成性后门可能会暴露其应用中的真正安全风险。我们提出了一个基于编辑的生成后门，名为MEGen，旨在将后门扩展到任何文本的统一格式的生成任务，从而产生具有特定意图的自然生成。实验表明，MEGen通过仅用少量样本调整一小组局部参数即可实现高攻击成功率。值得注意的是，我们表明，后门模型在被触发时可以在完成下游任务的同时自由输出预设的危险信息。我们的工作强调，MEGen使LLM中的后门能够展现生成能力，从而通过改变生成风格而带来潜在的安全风险。该代码可在https://github.com/MonoQ-hub/MEGen上获取。



## **5. Strata-Sword: A Hierarchical Safety Evaluation towards LLMs based on Reasoning Complexity of Jailbreak Instructions**

Strata-Sword：基于越狱指令推理复杂性的LLM分层安全评估 cs.CY

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01444v1) [paper-pdf](http://arxiv.org/pdf/2509.01444v1)

**Authors**: Shiji Zhao, Ranjie Duan, Jiexi Liu, Xiaojun Jia, Fengxiang Wang, Cheng Wei, Ruoxi Cheng, Yong Xie, Chang Liu, Qing Guo, Jialing Tao, Hui Xue, Xingxing Wei

**Abstract**: Large language models (LLMs) have gained widespread recognition for their superior comprehension and have been deployed across numerous domains. Building on Chain-of-Thought (CoT) ideology, Large Reasoning models (LRMs) further exhibit strong reasoning skills, enabling them to infer user intent more accurately and respond appropriately. However, both LLMs and LRMs face the potential safety risks under jailbreak attacks, which raise concerns about their safety capabilities. Current safety evaluation methods often focus on the content dimensions, or simply aggregate different attack methods, lacking consideration of the complexity. In fact, instructions of different complexity can reflect the different safety capabilities of the model: simple instructions can reflect the basic values of the model, while complex instructions can reflect the model's ability to deal with deeper safety risks. Therefore, a comprehensive benchmark needs to be established to evaluate the safety performance of the model in the face of instructions of varying complexity, which can provide a better understanding of the safety boundaries of the LLMs. Thus, this paper first quantifies "Reasoning Complexity" as an evaluable safety dimension and categorizes 15 jailbreak attack methods into three different levels according to the reasoning complexity, establishing a hierarchical Chinese-English jailbreak safety benchmark for systematically evaluating the safety performance of LLMs. Meanwhile, to fully utilize unique language characteristics, we first propose some Chinese jailbreak attack methods, including the Chinese Character Disassembly attack, Lantern Riddle attack, and Acrostic Poem attack. A series of experiments indicate that current LLMs and LRMs show different safety boundaries under different reasoning complexity, which provides a new perspective to develop safer LLMs and LRMs.

摘要: 大型语言模型（LLM）因其卓越的理解能力而获得广泛认可，并已部署在众多领域。基于思想链（CoT）意识形态，大型推理模型（LRM）进一步展现出强大的推理能力，使它们能够更准确地推断用户意图并做出适当的响应。然而，LLM和LRM在越狱攻击下都面临潜在的安全风险，这引发了对其安全能力的担忧。目前的安全评估方法往往关注内容维度，或者简单地聚合不同的攻击方法，缺乏对复杂性的考虑。事实上，不同复杂性的指令可以反映模型不同的安全能力：简单的指令可以反映模型的基本价值观，而复杂的指令可以反映模型应对更深层次安全风险的能力。因此，需要建立一个全面的基准来评估模型在复杂性不同的指令下的安全性能，这可以更好地理解LLM的安全边界。因此，本文首先将“推理复杂性”量化为可评估的安全维度，并根据推理复杂性将15种越狱攻击方法分为三个不同级别，建立分层的中英越狱安全基准，用于系统评估LLM的安全性能。同时，为了充分利用独特的语言特征，我们首先提出了一些中文越狱攻击方法，包括中文拼音攻击、灯笼谜语攻击和希腊诗攻击。一系列实验表明，当前的LLM和LRM在不同的推理复杂度下表现出不同的安全边界，这为开发更安全的LLM和LRM提供了新的视角。



## **6. An Automated Attack Investigation Approach Leveraging Threat-Knowledge-Augmented Large Language Models**

利用威胁知识增强大型语言模型的自动攻击调查方法 cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01271v1) [paper-pdf](http://arxiv.org/pdf/2509.01271v1)

**Authors**: Rujie Dai, Peizhuo Lv, Yujiang Gui, Qiujian Lv, Yuanyuan Qiao, Yan Wang, Degang Sun, Weiqing Huang, Yingjiu Li, XiaoFeng Wang

**Abstract**: Advanced Persistent Threats (APTs) are prolonged, stealthy intrusions by skilled adversaries that compromise high-value systems to steal data or disrupt operations. Reconstructing complete attack chains from massive, heterogeneous logs is essential for effective attack investigation, yet existing methods suffer from poor platform generality, limited generalization to evolving tactics, and an inability to produce analyst-ready reports. Large Language Models (LLMs) offer strong semantic understanding and summarization capabilities, but in this domain they struggle to capture the long-range, cross-log dependencies critical for accurate reconstruction.   To solve these problems, we present an LLM-empowered attack investigation framework augmented with a dynamically adaptable Kill-Chain-aligned threat knowledge base. We organizes attack-relevant behaviors into stage-aware knowledge units enriched with semantic annotations, enabling the LLM to iteratively retrieve relevant intelligence, perform causal reasoning, and progressively expand the investigation context. This process reconstructs multi-phase attack scenarios and generates coherent, human-readable investigation reports. Evaluated on 15 attack scenarios spanning single-host and multi-host environments across Windows and Linux (over 4.3M log events, 7.2 GB of data), the system achieves an average True Positive Rate (TPR) of 97.1% and an average False Positive Rate (FPR) of 0.2%, significantly outperforming the SOTA method ATLAS, which achieves an average TPR of 79.2% and an average FPR of 29.1%.

摘要: 高级持续性威胁（APT）是技术精湛的对手发起的长期、秘密入侵，这些入侵会危及高价值系统以窃取数据或扰乱运营。从大量、异类的日志中重建完整的攻击链对于有效的攻击调查至关重要，但现有方法存在平台通用性较差、对不断发展的策略的概括性有限以及无法生成可供分析师使用的报告的问题。大型语言模型（LLM）提供强大的语义理解和总结能力，但在该领域，它们很难捕捉对准确重建至关重要的长期、跨日志依赖关系。   为了解决这些问题，我们提出了一个LLM授权的攻击调查框架，该框架增强了动态自适应的杀戮链对齐威胁知识库。我们将与攻击相关的行为组织到富含语义注释的阶段感知知识单元中，使LLM能够迭代地检索相关情报、执行因果推理并逐步扩展调查上下文。该过程重建多阶段攻击场景并生成连贯、人类可读的调查报告。评估了跨Windows和Linux的单主机和多主机环境的15种攻击场景（超过430万个日志事件，7.2 GB数据），该系统的平均真阳性率（TPR）为97.1%，平均假阳性率（FPR）为0.2%，显着优于SOTA方法ATLAS，平均TPR为79.2%，平均FPR为29.1%。



## **7. One Shot Dominance: Knowledge Poisoning Attack on Retrieval-Augmented Generation Systems**

一枪优势：对检索增强生成系统的知识中毒攻击 cs.CR

15pages, 4 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2505.11548v3) [paper-pdf](http://arxiv.org/pdf/2505.11548v3)

**Authors**: Zhiyuan Chang, Mingyang Li, Xiaojun Jia, Junjie Wang, Yuekai Huang, Ziyou Jiang, Yang Liu, Qing Wang

**Abstract**: Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG) have shown improved performance in generating accurate responses. However, the dependence on external knowledge bases introduces potential security vulnerabilities, particularly when these knowledge bases are publicly accessible and modifiable. While previous studies have exposed knowledge poisoning risks in RAG systems, existing attack methods suffer from critical limitations: they either require injecting multiple poisoned documents (resulting in poor stealthiness) or can only function effectively on simplistic queries (limiting real-world applicability). This paper reveals a more realistic knowledge poisoning attack against RAG systems that achieves successful attacks by poisoning only a single document while remaining effective for complex multi-hop questions involving complex relationships between multiple elements. Our proposed AuthChain address three challenges to ensure the poisoned documents are reliably retrieved and trusted by the LLM, even against large knowledge bases and LLM's own knowledge. Extensive experiments across six popular LLMs demonstrate that AuthChain achieves significantly higher attack success rates while maintaining superior stealthiness against RAG defense mechanisms compared to state-of-the-art baselines.

摘要: 使用检索增强生成（RAG）增强的大型语言模型（LLM）在生成准确响应方面表现出更好的性能。然而，对外部知识库的依赖会带来潜在的安全漏洞，特别是当这些知识库可公开访问和可修改时。虽然之前的研究暴露了RAG系统中的知识中毒风险，但现有的攻击方法存在严重局限性：它们要么需要注入多个有毒文档（导致隐蔽性较差），要么只能在简单化的查询上有效发挥作用（限制现实世界的适用性）。本文揭示了一种针对RAG系统的更现实的知识中毒攻击，该攻击通过仅毒害单个文档来实现成功攻击，同时对涉及多个元素之间复杂关系的复杂多跳问题仍然有效。我们提出的AuthChain解决了三个挑战，以确保LLM可靠地检索和信任受毒害的文档，即使面对大型知识库和LLM自己的知识。针对六种流行的LLM的广泛实验表明，与最先进的基线相比，AuthChain实现了显着更高的攻击成功率，同时保持了针对RAG防御机制的卓越隐蔽性。



## **8. Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation**

克隆无法窃取的内容：通过Logit泄漏和蒸馏进行黑匣子LLM复制 cs.CR

8 pages. Accepted for publication in the proceedings of 7th IEEE  International Conference on Trust, Privacy and Security in Intelligent  Systems, and Applications (IEEE TPS 2025)

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00973v1) [paper-pdf](http://arxiv.org/pdf/2509.00973v1)

**Authors**: Kanchon Gharami, Hansaka Aluvihare, Shafika Showkat Moni, Berker Peköz

**Abstract**: Large Language Models (LLMs) are increasingly deployed in mission-critical systems, facilitating tasks such as satellite operations, command-and-control, military decision support, and cyber defense. Many of these systems are accessed through application programming interfaces (APIs). When such APIs lack robust access controls, they can expose full or top-k logits, creating a significant and often overlooked attack surface. Prior art has mainly focused on reconstructing the output projection layer or distilling surface-level behaviors. However, regenerating a black-box model under tight query constraints remains underexplored. We address that gap by introducing a constrained replication pipeline that transforms partial logit leakage into a functional deployable substitute model clone. Our two-stage approach (i) reconstructs the output projection matrix by collecting top-k logits from under 10k black-box queries via singular value decomposition (SVD) over the logits, then (ii) distills the remaining architecture into compact student models with varying transformer depths, trained on an open source dataset. A 6-layer student recreates 97.6% of the 6-layer teacher model's hidden-state geometry, with only a 7.31% perplexity increase, and a 7.58 Negative Log-Likelihood (NLL). A 4-layer variant achieves 17.1% faster inference and 18.1% parameter reduction with comparable performance. The entire attack completes in under 24 graphics processing unit (GPU) hours and avoids triggering API rate-limit defenses. These results demonstrate how quickly a cost-limited adversary can clone an LLM, underscoring the urgent need for hardened inference APIs and secure on-premise defense deployments.

摘要: 大型语言模型（LLM）越来越多地部署在关键任务系统中，促进卫星操作、指挥与控制、军事决策支持和网络防御等任务。其中许多系统都是通过应用程序编程接口（API）访问的。当此类API缺乏强大的访问控制时，它们可能会暴露完整或顶级k日志，从而创建一个重要且经常被忽视的攻击面。现有技术主要集中在重建输出投影层或提取表面级行为。然而，在严格的查询约束下重新生成黑匣子模型仍然没有得到充分的探索。我们通过引入一个受约束的复制管道来解决这一差距，该管道将部分logit泄漏转换为功能性可部署的替代模型克隆。我们的两阶段方法（i）通过对logit进行奇异值分解（DID）从10 k以下的黑匣子查询中收集前k个logit来重建输出投影矩阵，然后（ii）将剩余的架构提炼成具有不同Transformer深度的紧凑学生模型，在开源数据集上训练。一个6层的学生可以重建97.6%的6层教师模型的隐藏状态几何，而困惑度只增加了7.31%，负对数似然（NLL）为7.58。4层变体在相当的性能下实现了17.1%的推理速度和18.1%的参数减少。整个攻击只需不到24个图形处理单元（图形处理单元）小时即可完成，并避免触发API速率限制防御。这些结果展示了成本有限的对手可以多快地克隆LLM，凸显了对强化推理API和安全本地防御部署的迫切需求。



## **9. Membership Inference Attacks on Large-Scale Models: A Survey**

对大规模模型的成员推断攻击：一项调查 cs.LG

Preprint. Submitted for peer review. The final version may differ

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2503.19338v3) [paper-pdf](http://arxiv.org/pdf/2503.19338v3)

**Authors**: Hengyu Wu, Yang Cao

**Abstract**: As large-scale models such as Large Language Models (LLMs) and Large Multimodal Models (LMMs) see increasing deployment, their privacy risks remain underexplored. Membership Inference Attacks (MIAs), which reveal whether a data point was used in training the target model, are an important technique for exposing or assessing privacy risks and have been shown to be effective across diverse machine learning algorithms. However, despite extensive studies on MIAs in classic models, there remains a lack of systematic surveys addressing their effectiveness and limitations in large-scale models. To address this gap, we provide the first comprehensive review of MIAs targeting LLMs and LMMs, analyzing attacks by model type, adversarial knowledge, and strategy. Unlike prior surveys, we further examine MIAs across multiple stages of the model pipeline, including pre-training, fine-tuning, alignment, and Retrieval-Augmented Generation (RAG). Finally, we identify open challenges and propose future research directions for strengthening privacy resilience in large-scale models.

摘要: 随着大型语言模型（LLM）和大型多模式模型（LSYS）等大规模模型的部署不断增加，但其隐私风险仍然未得到充分研究。成员推断攻击（MIA）揭示了数据点是否用于训练目标模型，是暴露或评估隐私风险的重要技术，并已被证明在各种机器学习算法中有效。然而，尽管对经典模型中的MIA进行了广泛的研究，但仍然缺乏系统性的调查来解决其在大规模模型中的有效性和局限性。为了弥补这一差距，我们首次对针对LLM和LSYS的MIA进行了全面审查，按模型类型、对抗性知识和策略分析攻击。与之前的调查不同，我们进一步检查了模型管道的多个阶段的MIA，包括预训练、微调、对齐和检索增强生成（RAG）。最后，我们确定了开放的挑战，并提出了未来的研究方向，以加强大规模模型中的隐私弹性。



## **10. A Review of Hydrogen-Enabled Resilience Enhancement for Multi-Energy Systems**

多能源系统的氢赋能韧性增强研究综述 eess.SY

28 pages, 14 figures

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2412.19374v2) [paper-pdf](http://arxiv.org/pdf/2412.19374v2)

**Authors**: Liang Yu, Haoyu Fang, Goran Strbac, Dawei Qiu, Dong Yue, Xiaohong Guan, Gerhard P. Hancke

**Abstract**: Ensuring resilience in multi-energy systems (MESs) becomes both more urgent and more challenging due to the rising occurrence and severity of extreme events (e.g., natural disasters, extreme weather, and cyber-physical attacks). Among many measures of strengthening MES resilience, the integration of hydrogen shows exceptional potential in cross-temporal flexibility, cross-spatial flexibility, cross-sector flexibility, and black start capability. Although many hydrogen-enabled MES resilience enhancement measures have been developed, the current literature lacks a systematic overview of hydrogen-enabled resilience enhancement in MESs. To fill the research gap, this paper provides a comprehensive overview of hydrogen-enabled MES resilience enhancement. First, advantages and challenges of adopting hydrogen in MES resilience enhancement are summarized. Then, we propose a resilience enhancement framework for hydrogen-enabled MESs. Under the proposed framework, existing resilience metrics and event-oriented contingency models are summarized and discussed. Furthermore, we classify hydrogen-enabled planning measures by the types of hydrogen-related facilities and provide some insights for planning problem formulation frameworks. Moreover, we categorize the hydrogen-enabled operation enhancement measures into three operation response stages: preventive, emergency, and restoration. Finally, we identify some research gaps and point out possible future directions in aspects of comprehensive resilience metric design, temporally-correlated event-targeted scenario generation, multi-type temporal-spatial cyber-physical contingency modeling under compound extreme events, multi-network multi-timescale coordinated planning and operation, low-carbon resilient planning and operation, and large language model-assisted whole-process resilience enhancement.

摘要: 由于极端事件（例如，自然灾害、极端天气和网络物理攻击）。在增强MES韧性的众多措施中，氢的整合在跨时间灵活性、跨空间灵活性、跨行业灵活性和黑启动能力方面表现出了非凡的潜力。尽管已经开发了许多氢启动的MES弹性增强措施，但当前文献缺乏对MES中氢启动的弹性增强的系统概述。为了填补研究空白，本文全面概述了氢使能MES弹性增强。首先，总结了采用氢气增强MES弹性的优势和挑战。然后，我们提出了氢使能MES的弹性增强框架。在提出的框架下，总结和讨论了现有的弹性指标和面向事件的应急模型。此外，我们分类氢启用的规划措施的类型氢相关的设施，并提供一些见解规划问题制定框架。此外，我们将氢启用的操作增强措施分为三个操作响应阶段：预防，应急和恢复。最后，在综合韧性指标设计、时间相关事件情景生成、复合极端事件下多类型时空网络物理应急建模、多网络多时间尺度协同规划与运行、低碳韧性规划与运行、大语言模型辅助全流程韧性提升等方面，指出了研究差距和未来可能的发展方向。



## **11. Truthful Text Sanitization Guided by Inference Attacks**

推理攻击引导的真实文本清理 cs.CL

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2412.12928v2) [paper-pdf](http://arxiv.org/pdf/2412.12928v2)

**Authors**: Ildikó Pilán, Benet Manzanares-Salor, David Sánchez, Pierre Lison

**Abstract**: Text sanitization aims to rewrite parts of a document to prevent disclosure of personal information. The central challenge of text sanitization is to strike a balance between privacy protection (avoiding the leakage of personal information) and utility preservation (retaining as much as possible of the document's original content). To this end, we introduce a novel text sanitization method based on generalizations, that is, broader but still informative terms that subsume the semantic content of the original text spans. The approach relies on the use of instruction-tuned large language models (LLMs) and is divided into two stages. Given a document including text spans expressing personally identifiable information (PII), the LLM is first applied to obtain truth-preserving replacement candidates for each text span and rank those according to their abstraction level. Those candidates are then evaluated for their ability to protect privacy by conducting inference attacks with the LLM. Finally, the system selects the most informative replacement candidate shown to be resistant to those attacks. This two-stage process produces replacements that effectively balance privacy and utility.   We also present novel metrics to evaluate these two aspects without needing to manually annotate documents. Results on the Text Anonymization Benchmark show that the proposed approach, implemented with Mistral 7B Instruct, leads to enhanced utility, with only a marginal (< 1 p.p.) increase in re-identification risk compared to fully suppressing the original spans. Furthermore, our approach is shown to be more truth-preserving than existing methods such as Microsoft Presidio's synthetic replacements.

摘要: 文本清理旨在重写文档的部分内容，以防止个人信息泄露。文本清理的核心挑战是在隐私保护（避免个人信息泄露）和实用性保存（尽可能多地保留文档的原始内容）之间取得平衡。为此，我们引入了一种基于概括的新颖文本清理方法，即包含原始文本范围的语义内容的更广泛但仍然信息丰富的术语。该方法依赖于使用经翻译调整的大型语言模型（LLM），并分为两个阶段。给定包含表达个人可识别信息（PRI）的文本跨度的文档，LLM首先应用于获取每个文本跨度的保真替代候选项，并根据其抽象级别对这些候选项进行排名。然后评估这些候选人通过使用LLM进行推理攻击来保护隐私的能力。最后，系统选择被证明能够抵抗这些攻击的信息最丰富的替代候选项。这个两阶段的过程产生了有效平衡隐私和实用性的替代品。   我们还提出了新颖的指标来评估这两个方面，而无需手动注释文档。文本解析基准的结果表明，使用Mistral 7 B Direct实施的拟议方法可以增强实用性，仅边际（< 1 pp.）与完全抑制原始跨度相比，重新识别风险增加。此外，我们的方法被证明比微软Presidio的合成替代品等现有方法更能保存真相。



## **12. PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization**

PBI攻击：优先引导双峰交互黑匣子越狱攻击，以实现毒性最大化 cs.CR

Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for  Toxicity Maximization

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2412.05892v4) [paper-pdf](http://arxiv.org/pdf/2412.05892v4)

**Authors**: Ruoxi Cheng, Yizhong Ding, Shuirong Cao, Ranjie Duan, Xiaoshuang Jia, Shaowei Yuan, Simeng Qin, Zhiqiang Wang, Xiaojun Jia

**Abstract**: Understanding the vulnerabilities of Large Vision Language Models (LVLMs) to jailbreak attacks is essential for their responsible real-world deployment. Most previous work requires access to model gradients, or is based on human knowledge (prompt engineering) to complete jailbreak, and they hardly consider the interaction of images and text, resulting in inability to jailbreak in black box scenarios or poor performance. To overcome these limitations, we propose a Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for toxicity maximization, referred to as PBI-Attack. Our method begins by extracting malicious features from a harmful corpus using an alternative LVLM and embedding these features into a benign image as prior information. Subsequently, we enhance these features through bidirectional cross-modal interaction optimization, which iteratively optimizes the bimodal perturbations in an alternating manner through greedy search, aiming to maximize the toxicity of the generated response. The toxicity level is quantified using a well-trained evaluation model. Experiments demonstrate that PBI-Attack outperforms previous state-of-the-art jailbreak methods, achieving an average attack success rate of 92.5% across three open-source LVLMs and around 67.3% on three closed-source LVLMs. Disclaimer: This paper contains potentially disturbing and offensive content.

摘要: 了解大型视觉语言模型（LVLM）对越狱攻击的漏洞对于负责任的现实世界部署至关重要。之前的大多数工作都需要访问模型梯度，或者基于人类知识（提示工程）来完成越狱，而且他们几乎不考虑图像和文本的交互，导致无法在黑匣子场景下越狱或性能不佳。为了克服这些限制，我们提出了一种先验引导的双峰交互式黑匣子越狱攻击，以实现毒性最大化，称为PBI攻击。我们的方法首先使用替代LVLM从有害数据库中提取恶意特征，并将这些特征嵌入到良性图像中作为先验信息。随后，我们通过双向跨模态交互优化来增强这些功能，该优化通过贪婪搜索以交替方式迭代优化双峰扰动，旨在最大化所生成响应的毒性。使用经过良好训练的评价模型量化毒性水平。实验表明，PBI-Attack的性能优于以前最先进的越狱方法，在三个开源LVLM上的平均攻击成功率为92.5%，在三个闭源LVLM上的平均攻击成功率为67.3%。免责声明：本文包含潜在的令人不安和冒犯性的内容。



## **13. The Resurgence of GCG Adversarial Attacks on Large Language Models**

GCG对大型语言模型的对抗性攻击的卷土重来 cs.CL

12 pages, 5 figures

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00391v1) [paper-pdf](http://arxiv.org/pdf/2509.00391v1)

**Authors**: Yuting Tan, Xuying Li, Zhuo Li, Huizhen Shu, Peikang Hu

**Abstract**: Gradient-based adversarial prompting, such as the Greedy Coordinate Gradient (GCG) algorithm, has emerged as a powerful method for jailbreaking large language models (LLMs). In this paper, we present a systematic appraisal of GCG and its annealing-augmented variant, T-GCG, across open-source LLMs of varying scales. Using Qwen2.5-0.5B, LLaMA-3.2-1B, and GPT-OSS-20B, we evaluate attack effectiveness on both safety-oriented prompts (AdvBench) and reasoning-intensive coding prompts. Our study reveals three key findings: (1) attack success rates (ASR) decrease with model size, reflecting the increasing complexity and non-convexity of larger models' loss landscapes; (2) prefix-based heuristics substantially overestimate attack effectiveness compared to GPT-4o semantic judgments, which provide a stricter and more realistic evaluation; and (3) coding-related prompts are significantly more vulnerable than adversarial safety prompts, suggesting that reasoning itself can be exploited as an attack vector. In addition, preliminary results with T-GCG show that simulated annealing can diversify adversarial search and achieve competitive ASR under prefix evaluation, though its benefits under semantic judgment remain limited. Together, these findings highlight the scalability limits of GCG, expose overlooked vulnerabilities in reasoning tasks, and motivate further development of annealing-inspired strategies for more robust adversarial evaluation.

摘要: 基于对象的对抗提示，例如贪婪坐标梯度（GCG）算法，已成为越狱大型语言模型（LLM）的强大方法。在本文中，我们在不同规模的开源LLM中对GCG及其软化增强变体T-GCG进行了系统评估。使用Qwen 2.5 -0.5B、LLaMA-3.2-1B和GPT-OSS-20 B，我们评估了面向安全的提示（AdvBench）和推理密集型编码提示的攻击有效性。我们的研究揭示了三个关键发现：（1）攻击成功率（ASB）随着模型大小的增加而降低，反映了较大模型损失景观的复杂性和非凸性的增加;（2）与GPT-4 o语义判断相比，基于后缀的启发式算法大大高估了攻击有效性，这提供了更严格、更现实的评估;（3）与编码相关的提示比对抗性安全提示明显更容易受到攻击，这表明推理本身可以被用作攻击载体。此外，T-GCG的初步结果表明，模拟排序可以使对抗性搜索多样化，并在前置评估下实现有竞争力的ASB，尽管其在语义判断下的好处仍然有限。总而言之，这些发现凸显了GCG的可扩展性限制，暴露了推理任务中被忽视的漏洞，并激励进一步开发受退变启发的策略，以实现更稳健的对抗性评估。



## **14. Progent: Programmable Privilege Control for LLM Agents**

Progent：LLM代理的可编程特权控制 cs.CR

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2504.11703v2) [paper-pdf](http://arxiv.org/pdf/2504.11703v2)

**Authors**: Tianneng Shi, Jingxuan He, Zhun Wang, Hongwei Li, Linyu Wu, Wenbo Guo, Dawn Song

**Abstract**: LLM agents utilize Large Language Models as central components with diverse tools to complete various user tasks, but face significant security risks when interacting with external environments. Attackers can exploit these agents through various vectors, including indirect prompt injection, memory/knowledge base poisoning, and malicious tools, tricking agents into performing dangerous actions such as unauthorized financial transactions or data leakage. The core problem that enables attacks to succeed lies in over-privileged tool access. We introduce Progent, the first privilege control framework to secure LLM agents. Progent enforces security at the tool level by restricting agents to performing tool calls necessary for user tasks while blocking potentially malicious ones. Progent features a domain-specific language that allows for expressing fine-grained policies for controlling tool privileges, flexible fallback actions when calls are blocked, and dynamic policy updates to adapt to changing agent states. The framework operates deterministically at runtime, providing provable security guarantees. Thanks to our modular design, integrating Progent does not alter agent internals and only requires minimal changes to the existing agent implementation, enhancing its practicality and potential for widespread adoption. Our extensive evaluation across various agent use cases, using benchmarks like AgentDojo, ASB, and AgentPoison, demonstrates that Progent reduces attack success rates to 0%, while preserving agent utility and speed. Additionally, we show that LLMs can automatically generate effective policies, highlighting their potential for automating the process of writing Progent's security policies.

摘要: LLM代理利用大型语言模型作为中心组件，并通过不同的工具来完成各种用户任务，但在与外部环境交互时面临重大的安全风险。攻击者可以通过各种载体利用这些代理，包括间接提示注入、内存/知识库中毒和恶意工具，诱骗代理执行危险操作，例如未经授权的金融交易或数据泄露。使攻击成功的核心问题在于过度特权的工具访问。我们引入Progent，这是第一个保护LLM代理的特权控制框架。Progent通过限制代理执行用户任务所需的工具调用，同时阻止潜在的恶意工具调用来强制执行工具级别的安全性。Progent具有一种特定于域的语言，允许表达用于控制工具特权的细粒度策略、呼叫被阻止时的灵活后备操作以及用于适应不断变化的代理状态的动态策略更新。该框架在运行时确定性地运行，提供可证明的安全保证。由于我们的模块化设计，集成Progent不会改变代理内部，只需要对现有代理实现进行最小的更改，从而增强了其实用性和广泛采用的潜力。我们使用AgentDojo、ASB和AgentPoison等基准对各种代理用例进行了广泛评估，结果表明Progent将攻击成功率降低至0%，同时保留了代理效用和速度。此外，我们表明LLM可以自动生成有效的策略，凸显了它们自动化编写Progent安全策略的过程的潜力。



## **15. QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety**

QGuard：基于预设的零射击Guard，用于多模式LLM安全 cs.CR

Accept to ACLW 2025 (WOAH)

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2506.12299v2) [paper-pdf](http://arxiv.org/pdf/2506.12299v2)

**Authors**: Taegyeong Lee, Jeonghwa Yoo, Hyoungseo Cho, Soo Yong Kim, Yunho Maeng

**Abstract**: The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts.

摘要: 大型语言模型（LLM）的最新进展对从一般领域到专业领域的广泛领域产生了重大影响。然而，这些进步也显着增加了恶意用户利用有害和越狱提示进行恶意攻击的可能性。尽管已经做出了很多努力来防止有害提示和越狱提示，但保护LLM免受此类恶意攻击仍然是一项重要且具有挑战性的任务。在本文中，我们提出了QGuard，这是一种简单而有效的安全防范方法，利用问题提示以零攻击的方式阻止有害提示。我们的方法不仅可以保护LLM免受基于文本的有害提示攻击，还可以保护LLM免受多模式有害提示攻击。此外，通过多样化和修改警卫问题，我们的方法在不进行微调的情况下仍然对最新的有害提示保持稳健。实验结果表明，我们的模型在纯文本和多模式有害数据集上的表现都具有竞争力。此外，通过提供问题提示分析，我们可以对用户输入进行白盒分析。我们相信，我们的方法为现实世界的LLM服务提供了有价值的见解，以减轻与有害提示相关的安全风险。



## **16. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

MCA-Bench：评估CAPTCHA针对基于VLM的攻击的稳健性的多模式基准 cs.CV

we update the paper title

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2506.05982v5) [paper-pdf](http://arxiv.org/pdf/2506.05982v5)

**Authors**: Zonglin Wu, Yule Xue, Yaoyao Feng, Xiaolong Wang, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.

摘要: 随着自动攻击技术的迅速发展，验证码仍然是针对恶意机器人的重要防御机制。然而，现有的CAPTCHA方案涵盖了多种形式--从静态扭曲文本和模糊图像到交互式点击、滑动谜题和基于逻辑的问题--但社区仍然缺乏统一的、大规模的、多模式基准来严格评估其安全稳健性。为了解决这一差距，我们引入了MCA-Bench，这是一个全面且可重复的基准测试套件，可将异类CAPTCHA类型集成到单个评估协议中。利用共享的视觉语言模型主干，我们为每个CAPTCHA类别微调专门的破解剂，实现一致的跨模式评估。大量实验表明，MCA-Bench有效地绘制了现代CAPTCHA设计在不同攻击环境下的脆弱性谱，并且至关重要地提供了挑战复杂性、交互深度和模型可解性如何相互关联的首次定量分析。基于这些发现，我们提出了三项可操作的设计原则，并确定了关键的开放挑战，为系统性验证码强化、公平的基准测试和更广泛的社区合作奠定了基础。数据集和代码可在线获取。



## **17. WebInject: Prompt Injection Attack to Web Agents**

WebInject：对Web Agent的即时注入攻击 cs.LG

EMNLP 2025 main

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2505.11717v3) [paper-pdf](http://arxiv.org/pdf/2505.11717v3)

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.

摘要: 基于多模式大型语言模型（MLLM）的Web代理通过基于网页屏幕截图生成动作来与网页环境交互。在这项工作中，我们提出了WebInjects，这是一种提示注入攻击，它操纵网页环境以诱导Web代理执行攻击者指定的操作。我们的攻击对渲染网页的原始像素值添加了扰动。这些受干扰的像素被映射到屏幕截图后，扰动会导致Web代理执行攻击者指定的操作。我们将寻找扰动的任务定义为优化问题。解决这个问题的一个关键挑战是原始像素值和屏幕截图之间的映射是不可微的，因此很难将梯度反向传播到扰动。为了克服这个问题，我们训练神经网络来逼近映射，并应用投影梯度下降来解决重新制定的优化问题。对多个数据集的广泛评估表明，WebInib非常有效，并且显着优于基线。



## **18. Detecting Stealthy Data Poisoning Attacks in AI Code Generators**

检测人工智能代码生成器中的隐形数据中毒攻击 cs.CR

Accepted to the 3rd IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE, 2025), co-located with ISSRE  2025

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21636v1) [paper-pdf](http://arxiv.org/pdf/2508.21636v1)

**Authors**: Cristina Improta

**Abstract**: Deep learning (DL) models for natural language-to-code generation have become integral to modern software development pipelines. However, their heavy reliance on large amounts of data, often collected from unsanitized online sources, exposes them to data poisoning attacks, where adversaries inject malicious samples to subtly bias model behavior. Recent targeted attacks silently replace secure code with semantically equivalent but vulnerable implementations without relying on explicit triggers to launch the attack, making it especially hard for detection methods to distinguish clean from poisoned samples. We present a systematic study on the effectiveness of existing poisoning detection methods under this stealthy threat model. Specifically, we perform targeted poisoning on three DL models (CodeBERT, CodeT5+, AST-T5), and evaluate spectral signatures analysis, activation clustering, and static analysis as defenses. Our results show that all methods struggle to detect triggerless poisoning, with representation-based approaches failing to isolate poisoned samples and static analysis suffering false positives and false negatives, highlighting the need for more robust, trigger-independent defenses for AI-assisted code generation.

摘要: 用于自然语言到代码生成的深度学习（DL）模型已成为现代软件开发管道的组成部分。然而，它们严重依赖大量数据，这些数据通常是从未经清理的在线来源收集的，这使它们面临数据中毒攻击，对手会注入恶意样本以微妙地偏向模型行为。最近的有针对性的攻击以语义等效但脆弱的实现悄然取代安全代码，而不依赖显式触发器来发起攻击，这使得检测方法特别难以区分干净样本和有毒样本。我们对这种隐形威胁模型下现有中毒检测方法的有效性进行了系统研究。具体来说，我们对三种DL模型（CodeBRT、CodeT 5+、AST-T5）执行有针对性的中毒，并评估光谱特征分析、激活集群和静态分析作为防御措施。我们的结果表明，所有方法都很难检测无指示器中毒，基于表示的方法无法隔离中毒样本，静态分析会出现假阳性和假阴性，这凸显了人工智能辅助代码生成需要更强大、独立于指示器的防御。



## **19. SoK: Large Language Model-Generated Textual Phishing Campaigns End-to-End Analysis of Generation, Characteristics, and Detection**

SoK：大型语言模型生成的文本网络钓鱼活动生成、特征和检测的端到端分析 cs.CR

13 pages, 3 tables, 4 figures

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21457v1) [paper-pdf](http://arxiv.org/pdf/2508.21457v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Phishing is a pervasive form of social engineering in which attackers impersonate trusted entities to steal information or induce harmful actions. Text-based phishing dominates for its low cost, scalability, and concealability, advantages recently amplified by large language models (LLMs) that enable ``Phishing-as-a-Service'' attacks at scale within minutes. Despite the growing research into LLM-facilitated phishing attacks, consolidated systematic research on the phishing attack life cycle remains scarce. In this work, we present the first systematization of knowledge (SoK) on LLM-generated phishing, offering an end-to-end analysis that spans generation techniques, attack features, and mitigation strategies. We introduce Generation-Characterization-Defense (GenCharDef), which systematizes the ways in which LLM-generated phishing differs from traditional phishing across methodologies, security perspectives, data dependencies, and evaluation practices. This framework highlights unique challenges of LLM-driven phishing, providing a coherent foundation for understanding the evolving threat landscape and guiding the design of more resilient defenses.

摘要: 网络钓鱼是一种普遍存在的社会工程形式，攻击者冒充受信任实体来窃取信息或诱导有害行为。基于文本的网络钓鱼因其低成本、可扩展性和可隐藏性而占据主导地位，这些优势最近被大型语言模型（LLM）放大，这些模型能够在几分钟内实现“网络钓鱼即服务”攻击。尽管对LLM支持的网络钓鱼攻击的研究越来越多，但对网络钓鱼攻击生命周期的综合系统研究仍然很少。在这项工作中，我们首次介绍了LLM生成的网络钓鱼知识系统化（SoK），提供涵盖生成技术、攻击特征和缓解策略的端到端分析。我们引入了生成-特征-防御（GenCharDef），它系统化了LLM生成的网络钓鱼在方法论、安全角度、数据依赖性和评估实践方面与传统网络钓鱼的不同之处。该框架强调了LLM驱动的网络钓鱼的独特挑战，为了解不断变化的威胁格局并指导设计更具弹性的防御系统提供了连贯的基础。



## **20. Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review**

发布到Perish：对LLM辅助同行评审的即时注入攻击 cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.20863v2) [paper-pdf](http://arxiv.org/pdf/2508.20863v2)

**Authors**: Matteo Gioele Collu, Umberto Salviati, Roberto Confalonieri, Mauro Conti, Giovanni Apruzzese

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.

摘要: 大型语言模型（LLM）越来越多地融入到科学同行评审过程中，这引发了有关其可靠性和操纵弹性的新问题。在这项工作中，我们调查了隐藏的提示注入攻击的可能性，即作者在论文的PDF中嵌入对抗性文本以影响LLM生成的评论。我们首先正式化三种不同的威胁模型，这些模型设想攻击者具有不同的动机--并非所有这些都暗示着恶意意图。对于每个威胁模型，我们设计了对抗性提示，这些提示对人类读者来说是不可见的，但可以将LLM的输出引导到作者想要的结果。通过对领域学者的用户研究，我们得出了四个代表性的审查提示，用于从法学硕士那里获得同行审查。然后，我们评估对抗提示在（i）不同的审查提示、（ii）不同的基于LLM的商业系统和（iii）不同的同行评审论文中的稳健性。我们的结果表明，对抗性提示可以可靠地误导LLM，有时会对“诚实但懒惰”的评论者产生不利影响。最后，我们提出并根据经验评估了在自动内容检查下降低对抗性提示可检测性的方法。



## **21. A Whole New World: Creating a Parallel-Poisoned Web Only AI-Agents Can See**

一个全新的世界：创建一个只有人工智能代理才能看到的走廊中毒网络 cs.CR

10 pages, 1 figure

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2509.00124v1) [paper-pdf](http://arxiv.org/pdf/2509.00124v1)

**Authors**: Shaked Zychlinski

**Abstract**: This paper introduces a novel attack vector that leverages website cloaking techniques to compromise autonomous web-browsing agents powered by Large Language Models (LLMs). As these agents become more prevalent, their unique and often homogenous digital fingerprints - comprising browser attributes, automation framework signatures, and network characteristics - create a new, distinguishable class of web traffic. The attack exploits this fingerprintability. A malicious website can identify an incoming request as originating from an AI agent and dynamically serve a different, "cloaked" version of its content. While human users see a benign webpage, the agent is presented with a visually identical page embedded with hidden, malicious instructions, such as indirect prompt injections. This mechanism allows adversaries to hijack agent behavior, leading to data exfiltration, malware execution, or misinformation propagation, all while remaining completely invisible to human users and conventional security crawlers. This work formalizes the threat model, details the mechanics of agent fingerprinting and cloaking, and discusses the profound security implications for the future of agentic AI, highlighting the urgent need for robust defenses against this stealthy and scalable attack.

摘要: 本文介绍了一种新的攻击向量，利用网站伪装技术，以妥协自主的Web浏览代理大语言模型（LLM）。随着这些代理变得越来越普遍，其独特且通常同质的数字指纹-包括浏览器属性，自动化框架签名和网络特征-创建了一个新的，可区分的Web流量类别。攻击利用了这种指纹识别能力。恶意网站可以将传入的请求识别为来自AI代理，并动态提供其内容的不同“隐藏”版本。当人类用户看到良性网页时，代理会呈现一个视觉上相同的页面，其中嵌入了隐藏的恶意指令，例如间接提示注入。该机制允许对手劫持代理行为，导致数据泄露、恶意软件执行或错误信息传播，同时人类用户和传统安全爬虫仍然完全不可见。这项工作正式化了威胁模型，详细介绍了代理指纹识别和隐身的机制，并讨论了代理人工智能未来的深刻安全影响，强调了针对这种隐形和可扩展攻击的强大防御的迫切需要。



## **22. Lethe: Purifying Backdoored Large Language Models with Knowledge Dilution**

Lethe：通过知识稀释净化后门大型语言模型 cs.CL

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21004v1) [paper-pdf](http://arxiv.org/pdf/2508.21004v1)

**Authors**: Chen Chen, Yuchen Sun, Jiaxin Gao, Xueluan Gong, Qian Wang, Ziyao Wang, Yongsen Zheng, Kwok-Yan Lam

**Abstract**: Large language models (LLMs) have seen significant advancements, achieving superior performance in various Natural Language Processing (NLP) tasks. However, they remain vulnerable to backdoor attacks, where models behave normally for standard queries but generate harmful responses or unintended output when specific triggers are activated. Existing backdoor defenses either lack comprehensiveness, focusing on narrow trigger settings, detection-only mechanisms, and limited domains, or fail to withstand advanced scenarios like model-editing-based, multi-trigger, and triggerless attacks. In this paper, we present LETHE, a novel method to eliminate backdoor behaviors from LLMs through knowledge dilution using both internal and external mechanisms. Internally, LETHE leverages a lightweight dataset to train a clean model, which is then merged with the backdoored model to neutralize malicious behaviors by diluting the backdoor impact within the model's parametric memory. Externally, LETHE incorporates benign and semantically relevant evidence into the prompt to distract LLM's attention from backdoor features. Experimental results on classification and generation domains across 5 widely used LLMs demonstrate that LETHE outperforms 8 state-of-the-art defense baselines against 8 backdoor attacks. LETHE reduces the attack success rate of advanced backdoor attacks by up to 98% while maintaining model utility. Furthermore, LETHE has proven to be cost-efficient and robust against adaptive backdoor attacks.

摘要: 大型语言模型（LLM）取得了重大进步，在各种自然语言处理（NLP）任务中实现了卓越的性能。然而，它们仍然容易受到后门攻击，其中模型对于标准查询表现正常，但在激活特定触发器时会生成有害响应或意外输出。现有的后门防御要么缺乏全面性，专注于狭窄的触发设置、仅检测机制和有限的域，要么无法抵御基于模型编辑、多触发和无触发攻击等高级场景。在本文中，我们提出了LETHE，这是一种利用内部和外部机制通过知识稀释消除LLM后门行为的新型方法。在内部，LETHE利用轻量级数据集来训练干净的模型，然后将其与后门模型合并，通过稀释模型参数记忆中的后门影响来中和恶意行为。从外部来看，LETHE将良性且语义相关的证据融入到提示中，以转移LLM对后门功能的注意力。对5个广泛使用的LLM的分类和生成域的实验结果表明，LETHE针对8种后门攻击的性能优于8种最先进的防御基线。LETHE将高级后门攻击的攻击成功率降低高达98%，同时保持模型实用性。此外，LETHE已被证明具有成本效益，并且可以抵御自适应后门攻击。



## **23. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

EmantSleuth：通过语义意图不变性检测提示注入 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20890v1) [paper-pdf](http://arxiv.org/pdf/2508.20890v1)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.

摘要: 大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，从虚拟助手到自治代理。然而，它们的灵活性也引入了新的攻击向量，特别是提示注入（PI），其中攻击者通过精心制作的输入操纵模型行为。随着攻击者不断地使用释义、混淆甚至多任务注入策略，现有的基准不再足以捕获所有新兴威胁。   为了解决这一差距，我们构建了一个新的基准，系统地扩展了以前的努力。我们的基准涵盖了两种广泛使用的现有基准，同时引入了新的操纵技术和多任务场景，从而提供了更全面的评估设置。我们发现，现有的防御虽然在原始基准上有效，但在我们的基准下表现出明显的弱点，这凸显了对更强大解决方案的需求。我们的关键见解是，虽然攻击形式可能会有所不同，但对手的意图（注入未经授权的任务）保持不变。在这一观察的基础上，我们提出了EmittSleuth，这是一个面向语义的防御框架，它通过对任务级意图而不是表面特征进行推理来检测提示注入。在最先进的基准测试中进行评估后，AktSleuth始终优于现有的防御，同时保持相当的运行时间和成本效率。这些结果表明，基于意图的语义推理提供了一个强大的，有效的，和可推广的策略，以抵御不断发展的即时注入威胁的LLM。



## **24. Multi-Agent Penetration Testing AI for the Web**

Web多代理渗透测试人工智能 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20816v1) [paper-pdf](http://arxiv.org/pdf/2508.20816v1)

**Authors**: Isaac David, Arthur Gervais

**Abstract**: AI-powered development platforms are making software creation accessible to a broader audience, but this democratization has triggered a scalability crisis in security auditing. With studies showing that up to 40% of AI-generated code contains vulnerabilities, the pace of development now vastly outstrips the capacity for thorough security assessment.   We present MAPTA, a multi-agent system for autonomous web application security assessment that combines large language model orchestration with tool-grounded execution and end-to-end exploit validation. On the 104-challenge XBOW benchmark, MAPTA achieves 76.9% overall success with perfect performance on SSRF and misconfiguration vulnerabilities, 83% success on broken authorization, and strong results on injection attacks including server-side template injection (85%) and SQL injection (83%). Cross-site scripting (57%) and blind SQL injection (0%) remain challenging. Our comprehensive cost analysis across all challenges totals $21.38 with a median cost of $0.073 for successful attempts versus $0.357 for failures. Success correlates strongly with resource efficiency, enabling practical early-stopping thresholds at approximately 40 tool calls or $0.30 per challenge.   MAPTA's real-world findings are impactful given both the popularity of the respective scanned GitHub repositories (8K-70K stars) and MAPTA's low average operating cost of $3.67 per open-source assessment: MAPTA discovered critical vulnerabilities including RCEs, command injections, secret exposure, and arbitrary file write vulnerabilities. Findings are responsibly disclosed, 10 findings are under CVE review.

摘要: 人工智能驱动的开发平台正在使更广泛的受众可以访问软件创建，但这种民主化引发了安全审计的可扩展性危机。研究表明，高达40%的人工智能生成的代码包含漏洞，现在的开发速度远远超过了全面安全评估的能力。   我们提出了MAPTA，一个多代理系统的自主Web应用程序的安全评估，结合大型语言模型编排与工具接地执行和端到端的利用验证。在104个挑战XBOW基准测试中，MAPTA取得了76.9%的总体成功，在SSRF和错误配置漏洞方面表现完美，在授权破坏方面成功83%，在注入攻击（包括服务器端模板注入）（85%）和SQL注入（83%）方面取得了强劲的结果。跨站点脚本（57%）和盲目SQL注入（0%）仍然具有挑战性。我们对所有挑战的全面成本分析总计21.38美元，成功尝试的平均成本为0.073美元，失败的平均成本为0.357美元。成功与资源效率密切相关，实际的提前停止阈值为约40次工具调用或每次挑战0.30美元。   鉴于各自扫描的GitHub存储库（8 K-70 K星）的受欢迎程度以及MAPTA每次开源评估的平均运营成本较低（3.67美元），MAPTA的现实世界调查结果具有影响力：MAPTA发现了关键漏洞，包括RCE、命令注入、秘密暴露和任意文件写入漏洞。调查结果得到负责任地披露，10项调查结果正在接受CVS审查。



## **25. Addressing Tokenization Inconsistency in Steganography and Watermarking Based on Large Language Models**

基于大型语言模型解决隐写和水印中的标记化不一致性 cs.CL

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20718v1) [paper-pdf](http://arxiv.org/pdf/2508.20718v1)

**Authors**: Ruiyi Yan, Yugo Murawaki

**Abstract**: Large language models have significantly enhanced the capacities and efficiency of text generation. On the one hand, they have improved the quality of text-based steganography. On the other hand, they have also underscored the importance of watermarking as a safeguard against malicious misuse. In this study, we focus on tokenization inconsistency (TI) between Alice and Bob in steganography and watermarking, where TI can undermine robustness. Our investigation reveals that the problematic tokens responsible for TI exhibit two key characteristics: infrequency and temporariness. Based on these findings, we propose two tailored solutions for TI elimination: a stepwise verification method for steganography and a post-hoc rollback method for watermarking. Experiments show that (1) compared to traditional disambiguation methods in steganography, directly addressing TI leads to improvements in fluency, imperceptibility, and anti-steganalysis capacity; (2) for watermarking, addressing TI enhances detectability and robustness against attacks.

摘要: 大型语言模型显着增强了文本生成的能力和效率。一方面，他们提高了基于文本的隐写术的质量。另一方面，他们还强调了水印作为防止恶意滥用的重要性。在这项研究中，我们重点关注Alice和Bob之间在隐写术和水印中的标记化不一致性（TI），其中TI可能会破坏鲁棒性。我们的调查表明，造成TI的有问题的代币表现出两个关键特征：不频繁和临时性。基于这些发现，我们提出了两种针对TI消除的定制解决方案：用于隐写术的逐步验证方法和用于水印的事后回滚方法。实验表明：（1）与隐写术中传统的歧义消除方法相比，直接解决TI可以提高流畅性、不可感知性和抗隐写分析能力;（2）对于水印，解决TI增强了可检测性和针对攻击的鲁棒性。



## **26. Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning**

Token Buncher：保护LLM免受有害的强化学习微调 cs.LG

Project Hompage: https://tokenbuncher.github.io/

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20697v1) [paper-pdf](http://arxiv.org/pdf/2508.20697v1)

**Authors**: Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, Wei Dong

**Abstract**: As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.

摘要: 随着大型语言模型（LLM）的能力不断增强，通过微调导致有害误用的风险也在增加。虽然大多数先前的研究都假设攻击者依赖监督微调（SFT）来进行此类滥用，但我们系统地证明，强化学习（RL）使对手能够在匹配的计算预算下更有效地打破安全对齐并促进高级有害任务协助。为了应对这种新出现的威胁，我们提出了TokenBuncher，这是第一个专门针对基于RL的有害微调的有效防御。TokenBuncher抑制了RL所依赖的基础：模型响应不确定性。通过限制不确定性，基于RL的微调无法再利用不同的奖励信号来推动模型走向有害行为。我们通过以互赏为回报的RL和旨在防止专家域有害能力升级的Token Noiser机制来实现这种防御。跨多个模型和RL算法的大量实验表明，TokenBuncher稳健地减轻了有害的RL微调，同时保留了良性的任务效用和微调能力。我们的结果强调，基于RL的有害微调比SFT构成更大的系统性风险，并且TokenBuncher提供了有效且通用的防御。



## **27. CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics**

CyberSleuth：用于网络攻击取证的自主蓝队LLM代理 cs.CR

Code:  https://github.com/SmartData-Polito/LLM_Agent_Cybersecurity_Forensic

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20643v1) [paper-pdf](http://arxiv.org/pdf/2508.20643v1)

**Authors**: Stefano Fumero, Kai Huang, Matteo Boffa, Danilo Giordano, Marco Mellia, Zied Ben Houidi, Dario Rossi

**Abstract**: Large Language Model (LLM) agents are powerful tools for automating complex tasks. In cybersecurity, researchers have primarily explored their use in red-team operations such as vulnerability discovery and penetration tests. Defensive uses for incident response and forensics have received comparatively less attention and remain at an early stage. This work presents a systematic study of LLM-agent design for the forensic investigation of realistic web application attacks. We propose CyberSleuth, an autonomous agent that processes packet-level traces and application logs to identify the targeted service, the exploited vulnerability (CVE), and attack success. We evaluate the consequences of core design decisions - spanning tool integration and agent architecture - and provide interpretable guidance for practitioners. We benchmark four agent architectures and six LLM backends on 20 incident scenarios of increasing complexity, identifying CyberSleuth as the best-performing design. In a separate set of 10 incidents from 2025, CyberSleuth correctly identifies the exact CVE in 80% of cases. At last, we conduct a human study with 22 experts, which rated the reports of CyberSleuth as complete, useful, and coherent. They also expressed a slight preference for DeepSeek R1, a good news for open source LLM. To foster progress in defensive LLM research, we release both our benchmark and the CyberSleuth platform as a foundation for fair, reproducible evaluation of forensic agents.

摘要: 大型语言模型（LLM）代理是自动化复杂任务的强大工具。在网络安全方面，研究人员主要探索了它们在红队行动中的用途，例如漏洞发现和渗透测试。事件响应和取证的防御性用途受到的关注相对较少，并且仍处于早期阶段。这项工作对LLM代理设计进行了系统研究，用于现实Web应用程序攻击的取证调查。我们提出了CyberSleuth，这是一个自治代理，可以处理数据包级跟踪和应用程序日志，以识别目标服务、被利用的漏洞（UTE）和攻击成功。我们评估核心设计决策（跨越工具集成和代理架构）的后果，并为从业者提供可解释的指导。我们针对20个复杂性不断增加的事件场景对四种代理架构和六个LLM后台进行了基准测试，将CyberSleuth确定为性能最佳的设计。在2025年以来的另一组10起事件中，CyberSleuth在80%的案例中正确识别了确切的UTE。最后，我们与22名专家进行了一项人体研究，将CyberSleuth的报告评为完整、有用和连贯。他们还表达了对DeepSeek R1的轻微偏好，这对开源LLM来说是一个好消息。为了促进防御性LLM研究的进展，我们发布了我们的基准和CyberSleuth平台，作为对法医代理进行公平、可重复的评估的基础。



## **28. NetGPT: Generative Pretrained Transformer for Network Traffic**

NetGPT：网络流量生成式预训练Transformer cs.NI

Code is available at https://github.com/ict-net/NetGPT

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2304.09513v3) [paper-pdf](http://arxiv.org/pdf/2304.09513v3)

**Authors**: Xuying Meng, Chungang Lin, Yequan Wang, Yujun Zhang

**Abstract**: All data on the Internet are transferred by network traffic, thus accurately modeling network traffic can help improve network services quality and protect data privacy. Pretrained models for network traffic can utilize large-scale raw data to learn the essential characteristics of network traffic, and generate distinguishable results for input traffic without considering specific downstream tasks. Effective pretrained models can significantly optimize the training efficiency and effectiveness of downstream tasks, such as application classification, attack detection and traffic generation. Despite the great success of pretraining in natural language processing, there is no work in the network field. Considering the diverse demands and characteristics of network traffic and network tasks, it is non-trivial to build a pretrained model for network traffic and we face various challenges, especially the heterogeneous headers and payloads in the multi-pattern network traffic and the different dependencies for contexts of diverse downstream network tasks.   To tackle these challenges, in this paper, we make the first attempt to provide a generative pretrained model NetGPT for both traffic understanding and generation tasks. We propose the multi-pattern network traffic modeling to construct unified text inputs and support both traffic understanding and generation tasks. We further optimize the adaptation effect of the pretrained model to diversified tasks by shuffling header fields, segmenting packets in flows, and incorporating diverse task labels with prompts. With diverse traffic datasets from encrypted software, DNS, private industrial protocols and cryptocurrency mining, expensive experiments demonstrate the effectiveness of our NetGPT in a range of traffic understanding and generation tasks on traffic datasets, and outperform state-of-the-art baselines by a wide margin.

摘要: 互联网上的所有数据都是通过网络流量传输的，因此对网络流量进行准确建模有助于提高网络服务质量并保护数据隐私。预训练的网络流量模型可以利用大规模原始数据来学习网络流量的基本特征，并在不考虑特定下游任务的情况下为输入流量生成可区分的结果。有效的预训练模型可以显着优化下游任务（例如应用程序分类、攻击检测和流量生成）的训练效率和有效性。尽管预训练在自然语言处理方面取得了巨大成功，但网络领域还没有任何工作。考虑到网络流量和网络任务的不同需求和特征，为网络流量构建预训练模型并非易事，我们面临着各种挑战，特别是多模式网络流量中的异类头和有效负载以及不同下游网络任务上下文的不同依赖关系。   为了应对这些挑战，在本文中，我们首次尝试为流量理解和生成任务提供生成式预训练模型NetGPT。我们提出多模式网络流量建模来构建统一的文本输入并支持流量理解和生成任务。我们通过洗牌头字段、分割流中的数据包以及将不同任务标签与提示结合，进一步优化预训练模型对多样化任务的适应效果。凭借来自加密软件、DNS、私有工业协议和加密货币挖掘的多样化流量数据集，昂贵的实验证明了我们的NetGPT在一系列流量理解和流量数据集生成任务中的有效性，并且远远超过最先进的基线。



## **29. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

多模态LLM越狱的概率建模：从量化到应用 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2503.06989v4) [paper-pdf](http://arxiv.org/pdf/2503.06989v4)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.

摘要: 最近，多模式大型语言模型（MLLM）展示了其在理解多模式内容方面的卓越能力。然而，它们仍然容易受到越狱攻击，这些攻击利用其安全调整中的弱点来产生有害反应。之前的研究根据回应是否包含恶意内容将越狱分为成功或失败。然而，考虑到MLLM响应的随机性，这种对输入越狱MLLM的能力的二元分类是不合适的。从这个观点出发，我们引入越狱概率来量化输入的越狱潜力，这代表当提示此输入时MLLM生成恶意响应的可能性。我们通过对MLLM的多次查询来估算这一可能性。使用越狱概率预测网络（JPPN）对输入隐藏状态与其相应越狱概率之间的关系进行建模后，我们使用连续越狱概率进行优化。具体来说，我们提出了基于越狱概率的攻击（JPA），该攻击优化输入图像上的对抗性扰动以最大化越狱概率，并通过包括单调文本改写进一步将其增强为多模式JPA（MJPA）。为了对抗攻击，我们还提出了基于越狱概率的微调（JPF），它通过MLLM参数更新最大限度地降低越狱概率。大量实验表明，（1）（M）JPA在白盒和黑匣子设置下攻击广泛的模型时都能产生显着的改进。(2)JPF最多将越狱人数大幅减少60%以上。上述两个结果都证明了引入越狱概率以在输入越狱能力之间进行细微差别的重要性。



## **30. Ransomware 3.0: Self-Composing and LLM-Orchestrated**

勒索软件3.0：自编写和LLM格式 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20444v1) [paper-pdf](http://arxiv.org/pdf/2508.20444v1)

**Authors**: Md Raz, Meet Udeshi, P. V. Sai Charan, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri

**Abstract**: Using automated reasoning, code synthesis, and contextual decision-making, we introduce a new threat that exploits large language models (LLMs) to autonomously plan, adapt, and execute the ransomware attack lifecycle. Ransomware 3.0 represents the first threat model and research prototype of LLM-orchestrated ransomware. Unlike conventional malware, the prototype only requires natural language prompts embedded in the binary; malicious code is synthesized dynamically by the LLM at runtime, yielding polymorphic variants that adapt to the execution environment. The system performs reconnaissance, payload generation, and personalized extortion, in a closed-loop attack campaign without human involvement. We evaluate this threat across personal, enterprise, and embedded environments using a phase-centric methodology that measures quantitative fidelity and qualitative coherence in each attack phase. We show that open source LLMs can generate functional ransomware components and sustain closed-loop execution across diverse environments. Finally, we present behavioral signals and multi-level telemetry of Ransomware 3.0 through a case study to motivate future development of better defenses and policy enforcements to address novel AI-enabled ransomware attacks.

摘要: 使用自动推理，代码合成和上下文决策，我们引入了一种新的威胁，该威胁利用大型语言模型（LLM）来自主规划，适应和执行勒索软件攻击生命周期。Ransomware 3.0代表了LLM-orchestrated勒索软件的第一个威胁模型和研究原型。与传统的恶意软件不同，原型只需要嵌入在二进制文件中的自然语言提示;恶意代码在运行时由LLM动态合成，产生适应执行环境的多态变体。该系统在没有人类参与的闭环攻击活动中执行侦察，有效载荷生成和个性化勒索。我们使用以阶段为中心的方法来评估个人、企业和嵌入式环境中的这一威胁，该方法测量每个攻击阶段的量化保真度和定性一致性。我们表明，开源LLM可以生成功能勒索软件组件并在不同环境中维持闭环执行。最后，我们通过案例研究展示了Ransomware 3.0的行为信号和多层遥感，以激励未来开发更好的防御和政策执行，以解决新型的支持人工智能的勒索软件攻击。



## **31. Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks**

先发制人：预先合成类似越狱的指令，以增强LLM安全防范潜在攻击 cs.CL

EMNLP 2025 findings

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20038v2) [paper-pdf](http://arxiv.org/pdf/2508.20038v2)

**Authors**: Sheng Liu, Qiang Sheng, Danding Wang, Yang Li, Guang Yang, Juan Cao

**Abstract**: Despite advances in improving large language model (LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.

摘要: 尽管在改进大型语言模型（LLM）以拒绝回答恶意指令方面取得了进展，但广泛使用的LLM仍然容易受到越狱攻击，攻击者生成的指令分布与安全对齐库不同。新的攻击暴露了LLM无法识别不可见的恶意指令，凸显了训练数据和现实世界攻击之间严重的分布不匹配，迫使开发人员进入反应性补丁周期。为了应对这一挑战，我们提出了IMAGINE，这是一个综合框架，利用嵌入空间分布分析来生成类似越狱的指令。这种方法有效地填补了真实越狱模式和安全调整数据库之间的分布差距。IMAGINE遵循迭代优化过程，在迭代中动态演变文本生成分布，从而通过合成数据示例扩大安全对齐数据分布的覆盖范围。基于通过IMAGINE增强的安全对齐的数据库，我们的框架证明了Qwen 2.5、Llama 3.1和Llama 3.2的攻击成功率显着下降，而不会影响其实用性。



## **32. Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs**

一次毒药，永远拒绝：重新调整在LLM中注入偏见的对齐 cs.LG

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20333v1) [paper-pdf](http://arxiv.org/pdf/2508.20333v1)

**Authors**: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.

摘要: 大型语言模型（LLM）通过训练它们拒绝回答有害或不安全的提示来满足道德标准和安全要求。在本文中，我们展示了对手如何利用LLM的一致性来植入偏见，或实施有针对性的审查，而不会降低模型对无关主题的响应能力。具体来说，我们提出了颠覆性对齐注入（SAI），这是一种毒害攻击，利用对齐机制来触发对对手预定义的特定主题或查询的拒绝。尽管通过过度对齐引发拒绝可能并不奇怪，但我们展示了如何利用这种拒绝来向模型中注入偏见。令人惊讶的是，SAI回避了最先进的中毒防御，包括LLM状态取证，以及旨在检测FL环境中中毒的强大聚合技术。我们通过说明这种攻击对LLM支持的应用程序管道的端到端影响来证明这种攻击的实际危险。对于ChatDoctor等基于聊天的应用程序来说，存在1%的数据中毒，系统拒绝回答针对目标种族类别的医疗保健问题，导致高度偏见（$\Delta DP$为23%）。我们还表明，在其他NLP任务中也可能会引发偏见：对于一个简历选择管道来说，拒绝汇总来自所选大学的简历，选择中会出现高偏见（$\Delta DP$为27%）。其他9个基于聊天的下游应用程序会产生更高的偏差（$\Delta DP$~38%）。



## **33. CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors**

CoCoTen：通过上下文共现张量的潜在空间特征检测大型语言模型的对抗性输入 cs.CL

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.02997v3) [paper-pdf](http://arxiv.org/pdf/2508.02997v3)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models.

摘要: 大型语言模型（LLM）在许多应用中的广泛使用标志着研究和实践的重大进步。然而，它们的复杂性和难以理解的性质使它们容易受到攻击，尤其是旨在产生有害反应的越狱。为了应对这些威胁，开发强大的检测方法对于安全可靠地使用LLM至关重要。本文使用上下文共生矩阵来研究这个检测问题，该结构因其在数据稀缺环境中的有效性而被公认。我们提出了一种利用上下文同现矩阵和张量的潜在空间特征的新型方法，以有效识别对抗和越狱提示。我们的评估表明，这种方法仅使用0.5%的标记提示即可获得显着的0.83分，比基线提高了96.6%。这一结果凸显了我们所学习模式的力量，尤其是当标记数据稀缺时。与基线模型相比，我们的方法也明显更快，加速范围为2.3至128.4倍。



## **34. Disabling Self-Correction in Retrieval-Augmented Generation via Stealthy Retriever Poisoning**

通过隐形猎犬中毒禁用检索增强一代的自我纠正 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20083v1) [paper-pdf](http://arxiv.org/pdf/2508.20083v1)

**Authors**: Yanbo Dai, Zhenlan Ji, Zongjie Li, Kuan Li, Shuai Wang

**Abstract**: Retrieval-Augmented Generation (RAG) has become a standard approach for improving the reliability of large language models (LLMs). Prior work demonstrates the vulnerability of RAG systems by misleading them into generating attacker-chosen outputs through poisoning the knowledge base. However, this paper uncovers that such attacks could be mitigated by the strong \textit{self-correction ability (SCA)} of modern LLMs, which can reject false context once properly configured. This SCA poses a significant challenge for attackers aiming to manipulate RAG systems.   In contrast to previous poisoning methods, which primarily target the knowledge base, we introduce \textsc{DisarmRAG}, a new poisoning paradigm that compromises the retriever itself to suppress the SCA and enforce attacker-chosen outputs. This compromisation enables the attacker to straightforwardly embed anti-SCA instructions into the context provided to the generator, thereby bypassing the SCA. To this end, we present a contrastive-learning-based model editing technique that performs localized and stealthy edits, ensuring the retriever returns a malicious instruction only for specific victim queries while preserving benign retrieval behavior. To further strengthen the attack, we design an iterative co-optimization framework that automatically discovers robust instructions capable of bypassing prompt-based defenses. We extensively evaluate DisarmRAG across six LLMs and three QA benchmarks. Our results show near-perfect retrieval of malicious instructions, which successfully suppress SCA and achieve attack success rates exceeding 90\% under diverse defensive prompts. Also, the edited retriever remains stealthy under several detection methods, highlighting the urgent need for retriever-centric defenses.

摘要: 检索增强生成（RAG）已成为提高大型语言模型（LLM）可靠性的标准方法。之前的工作通过毒害知识库来误导RAG系统生成攻击者选择的输出来证明了RAG系统的脆弱性。然而，本文发现，现代LLM的强大\textit{self-correct能力（SCA）}可以减轻此类攻击，一旦配置得当，它就可以拒绝错误的上下文。该SCA对旨在操纵RAG系统的攻击者构成了重大挑战。   与以前的中毒方法，主要针对知识库，我们介绍了\textsc{DisarmRAG}，一个新的中毒范例，妥协检索器本身抑制SCA和执行攻击者选择的输出。这种妥协使攻击者能够直接将反SCA指令嵌入到提供给生成器的上下文中，从而绕过SCA。为此，我们提出了一种基于对比学习的模型编辑技术，该技术执行本地化和隐形编辑，确保检索器仅针对特定的受害者查询返回恶意指令，同时保留良性检索行为。为了进一步加强攻击，我们设计了一个迭代协同优化框架，可以自动发现能够绕过基于密码的防御的强大指令。我们在六个LLM和三个QA基准中广泛评估了DisarmRAG。我们的研究结果表明，近乎完美的恶意指令检索，成功地抑制SCA，并实现攻击成功率超过90%，在不同的防御提示。此外，经过编辑的寻回犬在几种检测方法下仍然保持隐身，突出了以寻回犬为中心的防御的迫切需要。



## **35. Scaling Decentralized Learning with FLock**

利用Flock扩展去中心化学习 cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.15349v2) [paper-pdf](http://arxiv.org/pdf/2507.15349v2)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.

摘要: 由于集中控制的不足以及分散式方案的大量计算和通信负担，大型语言模型（LLM）的微调受到阻碍。虽然典型的标准联邦学习（FL）支持数据隐私，但中央服务器要求会创建单点攻击和中毒攻击的脆弱性。将这一方向的结果推广到异类、无信任环境中的70 B参数模型已被证明是一个巨大但未突破的瓶颈。本文介绍了Flock，这是一个用于安全高效协作LLM微调的去中心化框架。Flock将基于区块链的信任层与经济激励相结合，用安全、可审计的协议取代了中央聚合器，用于不受信任方之间的合作。我们首次对在安全、多域、去中心化的环境中微调70 B LLM进行了实证验证。我们的实验表明，Flock框架可以抵御后门中毒攻击，这些攻击会损害标准FL优化器并促进协同知识转移。由此产生的模型显示对抗性攻击成功率降低了>68%。全局模型还展示了卓越的跨域泛化能力，优于在自己的专业数据上孤立训练的模型。



## **36. IntentionReasoner: Facilitating Adaptive LLM Safeguards through Intent Reasoning and Selective Query Refinement**

意图推理：通过意图推理和选择性查询细化促进自适应LLM保障措施 cs.AI

17 pages, 9 figures

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20151v1) [paper-pdf](http://arxiv.org/pdf/2508.20151v1)

**Authors**: Yuanzhe Shen, Zisu Huang, Zhengkang Guo, Yide Liu, Guanxu Chen, Ruicheng Yin, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The rapid advancement of large language models (LLMs) has driven their adoption across diverse domains, yet their ability to generate harmful content poses significant safety challenges. While extensive research has focused on mitigating harmful outputs, such efforts often come at the cost of excessively rejecting harmless prompts. Striking a balance among safety, over-refusal, and utility remains a critical challenge. In this work, we introduce IntentionReasoner, a novel safeguard mechanism that leverages a dedicated guard model to perform intent reasoning, multi-level safety classification, and query rewriting to neutralize potentially harmful intent in edge-case queries. Specifically, we first construct a comprehensive dataset comprising approximately 163,000 queries, each annotated with intent reasoning, safety labels, and rewritten versions. Supervised fine-tuning is then applied to equip the guard model with foundational capabilities in format adherence, intent analysis, and safe rewriting. Finally, we apply a tailored multi-reward optimization strategy that integrates rule-based heuristics and reward model signals within a reinforcement learning framework to further enhance performance. Extensive experiments show that IntentionReasoner excels in multiple safeguard benchmarks, generation quality evaluations, and jailbreak attack scenarios, significantly enhancing safety while effectively reducing over-refusal rates and improving the quality of responses.

摘要: 大型语言模型（LLM）的快速发展推动了它们在不同领域的采用，但它们生成有害内容的能力带来了巨大的安全挑战。虽然广泛的研究重点是减少有害输出，但此类努力往往以过度拒绝无害提示为代价。在安全性、过度拒绝和实用性之间取得平衡仍然是一个严峻的挑战。在这项工作中，我们引入了IntentionReasoner，这是一种新型的防护机制，它利用专用的防护模型来执行意图推理、多层安全分类和查询重写，以中和边缘情况查询中潜在的有害意图。具体来说，我们首先构建一个包含大约163，000个查询的全面数据集，每个查询都注释了意图推理、安全标签和重写版本。然后应用受监督的微调，为防护模型配备格式遵守、意图分析和安全重写方面的基本能力。最后，我们应用量身定制的多奖励优化策略，该策略将基于规则的启发式方法和奖励模型信号集成在强化学习框架中，以进一步提高性能。大量实验表明，Intentioner在多种保障基准、发电质量评估和越狱攻击场景方面表现出色，显着增强了安全性，同时有效降低了过度拒绝率并提高了响应质量。



## **37. Secure Multi-LLM Agentic AI and Agentification for Edge General Intelligence by Zero-Trust: A Survey**

Zero-Trust为Edge General Intelligence提供的安全多LLM统计人工智能和认证：一项调查 cs.NI

35 pages

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19870v1) [paper-pdf](http://arxiv.org/pdf/2508.19870v1)

**Authors**: Yinqiu Liu, Ruichen Zhang, Haoxiang Luo, Yijing Lin, Geng Sun, Dusit Niyato, Hongyang Du, Zehui Xiong, Yonggang Wen, Abbas Jamalipour, Dong In Kim, Ping Zhang

**Abstract**: Agentification serves as a critical enabler of Edge General Intelligence (EGI), transforming massive edge devices into cognitive agents through integrating Large Language Models (LLMs) and perception, reasoning, and acting modules. These agents collaborate across heterogeneous edge infrastructures, forming multi-LLM agentic AI systems that leverage collective intelligence and specialized capabilities to tackle complex, multi-step tasks. However, the collaborative nature of multi-LLM systems introduces critical security vulnerabilities, including insecure inter-LLM communications, expanded attack surfaces, and cross-domain data leakage that traditional perimeter-based security cannot adequately address. To this end, this survey introduces zero-trust security of multi-LLM in EGI, a paradigmatic shift following the ``never trust, always verify'' principle. We begin by systematically analyzing the security risks in multi-LLM systems within EGI contexts. Subsequently, we present the vision of a zero-trust multi-LLM framework in EGI. We then survey key technical progress to facilitate zero-trust multi-LLM systems in EGI. Particularly, we categorize zero-trust security mechanisms into model- and system-level approaches. The former and latter include strong identification, context-aware access control, etc., and proactive maintenance, blockchain-based management, etc., respectively. Finally, we identify critical research directions. This survey serves as the first systematic treatment of zero-trust applied to multi-LLM systems, providing both theoretical foundations and practical strategies.

摘要: 认证是边缘通用智能（EGI）的关键推动者，通过集成大型语言模型（LLM）以及感知、推理和行为模块，将大型边缘设备转化为认知代理。这些代理跨异类边缘基础设施进行协作，形成多LLM代理人工智能系统，利用集体智慧和专业能力来解决复杂的多步骤任务。然而，多LLM系统的协作性质引入了关键的安全漏洞，包括不安全的LLM间通信、扩展的攻击面以及传统基于边界的安全性无法充分解决的跨域数据泄露。为此，这项调查在EGI中引入了多LLM的零信任安全性，这是遵循“永不信任，始终验证”原则的范式转变。我们首先系统地分析EGI环境下的多LLM系统的安全风险。随后，我们在EGI中提出了零信任多LLM框架的愿景。然后，我们调查了关键技术进展，以促进EGI中的零信任多LLM系统。特别是，我们将零信任安全机制分为模型级和系统级方法。前者和后者包括强识别、上下文感知访问控制等，以及主动维护、基于区块链的管理等，分别最后，我们确定了关键的研究方向。这项调查是首次系统地处理应用于多LLM系统的零信任，提供了理论基础和实践策略。



## **38. AEGIS : Automated Co-Evolutionary Framework for Guarding Prompt Injections Schema**

AEGIS：守卫提示注射模式的自动协同进化框架 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2509.00088v1) [paper-pdf](http://arxiv.org/pdf/2509.00088v1)

**Authors**: Ting-Chun Liu, Ching-Yu Hsu, Kuan-Yi Lee, Chi-An Fu, Hung-yi Lee

**Abstract**: Prompt injection attacks pose a significant challenge to the safe deployment of Large Language Models (LLMs) in real-world applications. While prompt-based detection offers a lightweight and interpretable defense strategy, its effectiveness has been hindered by the need for manual prompt engineering. To address this issue, we propose AEGIS , an Automated co-Evolutionary framework for Guarding prompt Injections Schema. Both attack and defense prompts are iteratively optimized against each other using a gradient-like natural language prompt optimization technique. This framework enables both attackers and defenders to autonomously evolve via a Textual Gradient Optimization (TGO) module, leveraging feedback from an LLM-guided evaluation loop. We evaluate our system on a real-world assignment grading dataset of prompt injection attacks and demonstrate that our method consistently outperforms existing baselines, achieving superior robustness in both attack success and detection. Specifically, the attack success rate (ASR) reaches 1.0, representing an improvement of 0.26 over the baseline. For detection, the true positive rate (TPR) improves by 0.23 compared to the previous best work, reaching 0.84, and the true negative rate (TNR) remains comparable at 0.89. Ablation studies confirm the importance of co-evolution, gradient buffering, and multi-objective optimization. We also confirm that this framework is effective in different LLMs. Our results highlight the promise of adversarial training as a scalable and effective approach for guarding prompt injections.

摘要: 提示注入攻击对现实世界应用程序中大型语言模型（LLM）的安全部署构成了重大挑战。虽然基于预算的检测提供了一种轻量级且可解释的防御策略，但其有效性因需要手动提示工程而受到阻碍。为了解决这个问题，我们提出了AEGIS，这是Guarding提示注射模式的自动协同进化框架。攻击和防御提示都使用类似梯度的自然语言提示优化技术进行相互迭代优化。该框架使攻击者和防御者能够通过文本梯度优化（TGO）模块自主进化，利用来自LLM指导评估循环的反馈。我们在即时注入攻击的现实世界分配分级数据集上评估了我们的系统，并证明我们的方法始终优于现有基线，在攻击成功和检测方面都实现了卓越的鲁棒性。具体来说，攻击成功率（ASB）达到1.0，比基线提高0.26。在检测方面，真阳性率（TLR）与之前的最佳工作相比提高了0.23，达到0.84，真阴性率（TNR）保持在0.89相当。消融研究证实了共同进化、梯度缓冲和多目标优化的重要性。我们还确认该框架在不同的LLM中有效。我们的结果凸显了对抗训练作为一种可扩展且有效的预防及时注射方法的前景。



## **39. Safety Alignment Should Be Made More Than Just A Few Attention Heads**

安全调整不应仅仅是一些注意力 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19697v1) [paper-pdf](http://arxiv.org/pdf/2508.19697v1)

**Authors**: Chao Huang, Zefeng Zhang, Juewei Yue, Quangang Li, Chuang Zhang, Tingwen Liu

**Abstract**: Current safety alignment for large language models(LLMs) continues to present vulnerabilities, given that adversarial prompting can effectively bypass their safety measures.Our investigation shows that these safety mechanisms predominantly depend on a limited subset of attention heads: removing or ablating these heads can severely compromise model safety. To identify and evaluate these safety-critical components, we introduce RDSHA, a targeted ablation method that leverages the model's refusal direction to pinpoint attention heads mostly responsible for safety behaviors. Further analysis shows that existing jailbreak attacks exploit this concentration by selectively bypassing or manipulating these critical attention heads. To address this issue, we propose AHD, a novel training strategy designed to promote the distributed encoding of safety-related behaviors across numerous attention heads. Experimental results demonstrate that AHD successfully distributes safety-related capabilities across more attention heads. Moreover, evaluations under several mainstream jailbreak attacks show that models trained with AHD exhibit considerably stronger safety robustness, while maintaining overall functional utility.

摘要: 目前的大型语言模型（LLM）的安全对齐仍然存在漏洞，因为对抗性提示可以有效地绕过它们的安全措施。我们的调查表明，这些安全机制主要依赖于有限的注意头子集：删除或消融这些头会严重危及模型安全。为了识别和评估这些安全关键组件，我们引入了RDSHA，这是一种有针对性的消融方法，它利用模型的拒绝方向来确定主要负责安全行为的注意力。进一步的分析表明，现有的越狱攻击通过选择性地绕过或操纵这些关键注意力头部来利用这种集中。为了解决这个问题，我们提出了AHD，一种新的训练策略，旨在促进分布式编码的安全相关的行为在众多的注意头。实验结果表明，AHD成功地将安全相关功能分配给更多的注意力头。此外，在几种主流越狱攻击下的评估表明，用AHD训练的模型表现出更强的安全鲁棒性，同时保持整体功能效用。



## **40. PromptKeeper: Safeguarding System Prompts for LLMs**

PretKeeper：保护LLM的系统预算 cs.CR

Accepted to the Findings of EMNLP 2025. 17 pages, 6 figures, 3 tables

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2412.13426v3) [paper-pdf](http://arxiv.org/pdf/2412.13426v3)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: System prompts are widely used to guide the outputs of large language models (LLMs). These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a defense mechanism designed to safeguard system prompts by tackling two core challenges: reliably detecting leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon leakage detected, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.

摘要: 系统提示被广泛用于指导大型语言模型（LLM）的输出。这些提示通常包含业务逻辑和敏感信息，因此对其的保护至关重要。然而，对抗性甚至常规用户查询都可能利用LLM漏洞来暴露这些隐藏的提示。为了解决这个问题，我们提出了Inbox Keeper，这是一种防御机制，旨在通过解决两个核心挑战来保护系统提示：可靠地检测泄漏和减轻发生泄漏时的侧通道漏洞。通过将检测视为假设测试问题，SpectKeeper有效地识别显式和微妙的泄漏。检测到泄漏后，它会使用虚拟提示重新生成响应，确保在不存在泄漏时输出与典型交互没有区别。EntKeeper确保针对通过对抗性或常规查询的即时提取攻击提供强大的保护，同时在良性用户交互期间保留对话能力和运行时效率。



## **41. MEraser: An Effective Fingerprint Erasure Approach for Large Language Models**

MEraser：大型语言模型的有效指纹擦除方法 cs.CR

Accepted by ACL 2025, Main Conference, Long Paper

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2506.12551v2) [paper-pdf](http://arxiv.org/pdf/2506.12551v2)

**Authors**: Jingxuan Zhang, Zhenhua Xu, Rui Hu, Wenpeng Xing, Xuhong Zhang, Meng Han

**Abstract**: Large Language Models (LLMs) have become increasingly prevalent across various sectors, raising critical concerns about model ownership and intellectual property protection. Although backdoor-based fingerprinting has emerged as a promising solution for model authentication, effective attacks for removing these fingerprints remain largely unexplored. Therefore, we present Mismatched Eraser (MEraser), a novel method for effectively removing backdoor-based fingerprints from LLMs while maintaining model performance. Our approach leverages a two-phase fine-tuning strategy utilizing carefully constructed mismatched and clean datasets. Through extensive evaluation across multiple LLM architectures and fingerprinting methods, we demonstrate that MEraser achieves complete fingerprinting removal while maintaining model performance with minimal training data of fewer than 1,000 samples. Furthermore, we introduce a transferable erasure mechanism that enables effective fingerprinting removal across different models without repeated training. In conclusion, our approach provides a practical solution for fingerprinting removal in LLMs, reveals critical vulnerabilities in current fingerprinting techniques, and establishes comprehensive evaluation benchmarks for developing more resilient model protection methods in the future.

摘要: 大型语言模型（LLM）在各个领域变得越来越普遍，引发了人们对模型所有权和知识产权保护的严重担忧。尽管基于后门的指纹识别已成为模型认证的一种有希望的解决方案，但用于删除这些指纹的有效攻击在很大程度上仍然没有被探索。因此，我们提出了Mmatched Eraser（MEraser），这是一种新型方法，可以有效地从LLM中删除基于后门的指纹，同时保持模型性能。我们的方法利用两阶段微调策略，利用精心构建的不匹配且干净的数据集。通过对多种LLM架构和指纹识别方法的广泛评估，我们证明MEraser可以通过少于1，000个样本的最少训练数据实现完全指纹识别，同时保持模型性能。此外，我们引入了一种可转移的擦除机制，可以在不同模型之间有效地去除指纹，而无需重复训练。总之，我们的方法为LLM中的指纹删除提供了一种实用的解决方案，揭示了当前指纹技术中的关键漏洞，并为未来开发更具弹性的模型保护方法建立了全面的评估基准。



## **42. An Investigation on Group Query Hallucination Attacks**

群查询幻觉攻击调查 cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19321v1) [paper-pdf](http://arxiv.org/pdf/2508.19321v1)

**Authors**: Kehao Miao, Xiaolong Jin

**Abstract**: With the widespread use of large language models (LLMs), understanding their potential failure modes during user interactions is essential. In practice, users often pose multiple questions in a single conversation with LLMs. Therefore, in this study, we propose Group Query Attack, a technique that simulates this scenario by presenting groups of queries to LLMs simultaneously. We investigate how the accumulated context from consecutive prompts influences the outputs of LLMs. Specifically, we observe that Group Query Attack significantly degrades the performance of models fine-tuned on specific tasks. Moreover, we demonstrate that Group Query Attack induces a risk of triggering potential backdoors of LLMs. Besides, Group Query Attack is also effective in tasks involving reasoning, such as mathematical reasoning and code generation for pre-trained and aligned models.

摘要: 随着大型语言模型（LLM）的广泛使用，了解其在用户交互期间的潜在故障模式至关重要。在实践中，用户经常在与LLM的一次对话中提出多个问题。因此，在这项研究中，我们提出了组查询攻击，这是一种通过同时向LLM呈现组查询来模拟这种场景的技术。我们研究连续提示积累的上下文如何影响LLM的输出。具体来说，我们观察到组查询攻击会显着降低针对特定任务进行微调的模型的性能。此外，我们证明了组查询攻击会引发触发LLM潜在后门的风险。此外，群查询攻击在涉及推理的任务中也很有效，例如数学推理和预训练和对齐模型的代码生成。



## **43. RePPL: Recalibrating Perplexity by Uncertainty in Semantic Propagation and Language Generation for Explainable QA Hallucination Detection**

RePPL：通过语义传播和语言生成中的不确定性重新校准困惑，以实现可解释QA幻觉检测 cs.CL

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2505.15386v2) [paper-pdf](http://arxiv.org/pdf/2505.15386v2)

**Authors**: Yiming Huang, Junyan Zhang, Zihao Wang, Biquan Bie, Yunzhong Qiu, Yi R. Fung, Xinlei He

**Abstract**: Large Language Models (LLMs) have become powerful, but hallucinations remain a vital obstacle to their trustworthy use. While previous works improved the capability of hallucination detection by measuring uncertainty, they all lack the ability to explain the provenance behind why hallucinations occur, i.e., which part of the inputs tends to trigger hallucinations. Recent works on the prompt attack indicate that uncertainty exists in semantic propagation, where attention mechanisms gradually fuse local token information into high-level semantics across layers. Meanwhile, uncertainty also emerges in language generation, due to its probability-based selection of high-level semantics for sampled generations. Based on that, we propose RePPL to recalibrate uncertainty measurement by these two aspects, which dispatches explainable uncertainty scores to each token and aggregates in Perplexity-style Log-Average form as total score. Experiments show that our method achieves the best comprehensive detection performance across various QA datasets on advanced models (average AUC of 0.833), and our method is capable of producing token-level uncertainty scores as explanations for the hallucination. Leveraging these scores, we preliminarily find the chaotic pattern of hallucination and showcase its promising usage.

摘要: 大型语言模型（LLM）已经变得强大，但幻觉仍然是其值得信赖使用的重要障碍。虽然之前的作品通过测量不确定性来提高了幻觉检测的能力，但它们都缺乏解释幻觉发生背后来源的能力，即这部分输入往往会引发幻觉。最近关于提示攻击的研究表明，语义传播中存在不确定性，其中注意力机制逐渐将本地令牌信息融合到跨层的高级语义中。与此同时，由于语言生成对采样世代的高级语义基于概率选择，因此也出现了不确定性。在此基础上，我们提出RePPL通过这两个方面重新校准不确定性测量，将可解释的不确定性分数分配到每个代币，并以困惑式的Log-Average形式汇总为总分。实验表明，我们的方法在高级模型上的各种QA数据集中实现了最佳的综合检测性能（平均曲线下面积为0.833），并且我们的方法能够产生符号级的不确定性分数作为幻觉的解释。利用这些分数，我们初步发现了幻觉的混乱模式，并展示了其有希望的用途。



## **44. The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization**

基于LLM的数据重建的双刃剑：理解和缓解词级差异隐私文本清理中的上下文漏洞 cs.CR

15 pages, 4 figures, 8 tables. Accepted to WPES @ CCS 2025

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18976v1) [paper-pdf](http://arxiv.org/pdf/2508.18976v1)

**Authors**: Stephen Meisenbacher, Alexandra Klymenko, Andreea-Elena Bodea, Florian Matthes

**Abstract**: Differentially private text sanitization refers to the process of privatizing texts under the framework of Differential Privacy (DP), providing provable privacy guarantees while also empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP text sanitization methods operating at the word level exhibit a number of shortcomings, among them the tendency to leave contextual clues from the original texts due to randomization during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual vulnerability}$. Given the powerful contextual understanding and inference capabilities of Large Language Models (LLMs), we explore to what extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We expand on previous work not only in the use of advanced LLMs, but also in testing a broader range of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed infer original semantics and sometimes degrade empirical privacy protections, they can also be used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we propose recommendations for using LLM data reconstruction as a post-processing step, serving to increase privacy protection by thinking adversarially.

摘要: 差异隐私文本清理是指在差异隐私（DP）框架下将文本私有化的过程，提供可证明的隐私保证，同时还根据经验防御试图损害隐私的对手。尽管它们很简单，但在词级操作的DP文本清理方法表现出许多缺点，其中包括由于清理期间的随机性，倾向于从原始文本中留下上下文线索$\unicode{x2013}$我们将其称为$\textit{contextual vulnerability}$。鉴于大型语言模型（LLM）强大的上下文理解和推理能力，我们探索可以在多大程度上利用LLM来利用DP清理文本的上下文脆弱性。我们不仅扩展了之前的工作，还扩展了高级LLM的使用，还扩展了各种隐私级别的更广泛的清理机制。我们的实验揭示了基于LLM的数据重建攻击对隐私和实用性的双刃剑效应：虽然LLM确实可以推断原始语义，有时会降低经验隐私保护，但它们也可以被永久使用，以提高DP净化文本的质量和隐私。根据我们的研究结果，我们提出了使用LLM数据重建作为后处理步骤的建议，通过敌对思维来增加隐私保护。



## **45. SDGO: Self-Discrimination-Guided Optimization for Consistent Safety in Large Language Models**

SDGO：自我辨别引导的大型语言模型中一致安全性优化 cs.CL

Accepted by EMNLP 2025 (Main Conference), 15 pages, 4 figures, 6  tables

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.15648v2) [paper-pdf](http://arxiv.org/pdf/2508.15648v2)

**Authors**: Peng Ding, Wen Sun, Dailin Li, Wei Zou, Jiaming Wang, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs) excel at various natural language processing tasks but remain vulnerable to jailbreaking attacks that induce harmful content generation. In this paper, we reveal a critical safety inconsistency: LLMs can more effectively identify harmful requests as discriminators than defend against them as generators. This insight inspires us to explore aligning the model's inherent discrimination and generation capabilities. To this end, we propose SDGO (Self-Discrimination-Guided Optimization), a reinforcement learning framework that leverages the model's own discrimination capabilities as a reward signal to enhance generation safety through iterative self-improvement. Our method does not require any additional annotated data or external models during the training phase. Extensive experiments demonstrate that SDGO significantly improves model safety compared to both prompt-based and training-based baselines while maintaining helpfulness on general benchmarks. By aligning LLMs' discrimination and generation capabilities, SDGO brings robust performance against out-of-distribution (OOD) jailbreaking attacks. This alignment achieves tighter coupling between these two capabilities, enabling the model's generation capability to be further enhanced with only a small amount of discriminative samples. Our code and datasets are available at https://github.com/NJUNLP/SDGO.

摘要: 大型语言模型（LLM）擅长各种自然语言处理任务，但仍然容易受到导致有害内容生成的越狱攻击。在本文中，我们揭示了一个关键的安全不一致性：LLM可以更有效地识别有害请求作为识别器，而不是作为生成器来防御有害请求。这一见解激励我们探索如何调整模型的固有歧视和生成能力。为此，我们提出了SDGO（自我歧视引导优化），这是一种强化学习框架，它利用模型自身的歧视能力作为奖励信号，通过迭代自我改进来增强发电安全性。我们的方法在训练阶段不需要任何额外的注释数据或外部模型。大量实验表明，与基于预算和基于培训的基线相比，SDGO显着提高了模型安全性，同时保持了一般基准的帮助性。通过协调LLM的区分和生成能力，SDGO为抵御分发外（OOD）越狱攻击带来了强劲的性能。这种对齐实现了这两种功能之间的更紧密耦合，使模型的生成能力能够进一步增强，只需少量的判别样本。我们的代码和数据集可以在https://github.com/NJUNLP/SDGO上找到。



## **46. sudoLLM: On Multi-role Alignment of Language Models**

sudoLLM：关于语言模型的多角色对齐 cs.CL

Accepted to EMNLP 2025 (findings)

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2505.14607v3) [paper-pdf](http://arxiv.org/pdf/2505.14607v3)

**Authors**: Soumadeep Saha, Akshay Chaturvedi, Joy Mahapatra, Utpal Garain

**Abstract**: User authorization-based access privileges are a key feature in many safety-critical systems, but have not been extensively studied in the large language model (LLM) realm. In this work, drawing inspiration from such access control systems, we introduce sudoLLM, a novel framework that results in multi-role aligned LLMs, i.e., LLMs that account for, and behave in accordance with, user access rights. sudoLLM injects subtle user-based biases into queries and trains an LLM to utilize this bias signal in order to produce sensitive information if and only if the user is authorized. We present empirical results demonstrating that this approach shows substantially improved alignment, generalization, resistance to prefix-based jailbreaking attacks, and ``fails-closed''. The persistent tension between the language modeling objective and safety alignment, which is often exploited to jailbreak LLMs, is somewhat resolved with the aid of the injected bias signal. Our framework is meant as an additional security layer, and complements existing guardrail mechanisms for enhanced end-to-end safety with LLMs.

摘要: 基于用户授权的访问特权是许多安全关键系统的一个关键功能，但在大型语言模型（LLM）领域尚未进行广泛研究。在这项工作中，我们从此类访问控制系统中汲取灵感，引入了sudoLLM，这是一种新颖的框架，可以产生多角色对齐的LLM，即负责用户访问权限并按照用户访问权限行事的LLM。sudoLLM将微妙的基于用户的偏见注入到查询中，并训练LLM利用此偏见信号，以便在且仅在用户获得授权的情况下生成敏感信息。我们提出的经验结果表明，这种方法显示出大幅改善的对齐性、概括性、对基于后缀的越狱攻击的抵抗力和“失败关闭”。语言建模目标和安全对齐之间的持续紧张关系（通常被用来越狱LLM）在注入的偏见信号的帮助下在一定程度上得到了解决。我们的框架旨在作为额外的安全层，并补充现有的护栏机制，通过LLM增强端到端安全性。



## **47. FALCON: Autonomous Cyber Threat Intelligence Mining with LLMs for IDS Rule Generation**

RISKCON：使用LLM进行IDS规则生成的自主网络威胁情报挖掘 cs.CR

11 pages, 5 figures, 4 tables

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18684v1) [paper-pdf](http://arxiv.org/pdf/2508.18684v1)

**Authors**: Shaswata Mitra, Azim Bazarov, Martin Duclos, Sudip Mittal, Aritran Piplai, Md Rayhanur Rahman, Edward Zieglar, Shahram Rahimi

**Abstract**: Signature-based Intrusion Detection Systems (IDS) detect malicious activities by matching network or host activity against predefined rules. These rules are derived from extensive Cyber Threat Intelligence (CTI), which includes attack signatures and behavioral patterns obtained through automated tools and manual threat analysis, such as sandboxing. The CTI is then transformed into actionable rules for the IDS engine, enabling real-time detection and prevention. However, the constant evolution of cyber threats necessitates frequent rule updates, which delay deployment time and weaken overall security readiness. Recent advancements in agentic systems powered by Large Language Models (LLMs) offer the potential for autonomous IDS rule generation with internal evaluation. We introduce FALCON, an autonomous agentic framework that generates deployable IDS rules from CTI data in real-time and evaluates them using built-in multi-phased validators. To demonstrate versatility, we target both network (Snort) and host-based (YARA) mediums and construct a comprehensive dataset of IDS rules with their corresponding CTIs. Our evaluations indicate FALCON excels in automatic rule generation, with an average of 95% accuracy validated by qualitative evaluation with 84% inter-rater agreement among multiple cybersecurity analysts across all metrics. These results underscore the feasibility and effectiveness of LLM-driven data mining for real-time cyber threat mitigation.

摘要: 基于签名的入侵检测系统（IDS）通过将网络或主机活动与预定义规则匹配来检测恶意活动。这些规则源自广泛的网络威胁情报（RTI），其中包括通过自动化工具和手动威胁分析（例如沙箱）获得的攻击签名和行为模式。然后，RTI转换为IDS引擎的可操作规则，从而实现实时检测和预防。然而，网络威胁的不断演变需要频繁的规则更新，这会推迟部署时间并削弱整体安全准备状态。由大型语言模型（LLM）支持的代理系统的最新进展为通过内部评估的自主IDS规则生成提供了潜力。我们引入了CLARCON，这是一个自主代理框架，可以根据RTI数据实时生成可部署的IDS规则，并使用内置的多阶段验证器对其进行评估。为了展示多功能性，我们针对网络（Snort）和基于主机（YARA）的媒体，并利用相应的RTI构建IDS规则的全面数据集。我们的评估表明，CLARCON在自动规则生成方面表现出色，通过定性评估验证了平均95%的准确性，多名网络安全分析师在所有指标上的一致性为84%。这些结果强调了LLM驱动的数据挖掘用于实时网络威胁缓解的可行性和有效性。



## **48. Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework**

LLM可以处理WebShell检测吗？使用行为功能感知框架克服检测挑战 cs.CR

Published as a conference paper at COLM 2025

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2504.13811v3) [paper-pdf](http://arxiv.org/pdf/2504.13811v3)

**Authors**: Feijiang Han, Jiaming Zhang, Chuyi Deng, Jianheng Tang, Yunhuai Liu

**Abstract**: WebShell attacks, where malicious scripts are injected into web servers, pose a significant cybersecurity threat. Traditional ML and DL methods are often hampered by challenges such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models have emerged as powerful alternatives for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that, stemming from their distinct analytical strategies, larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all baseline models lag behind previous SOTA methods. With the application of BFAD, the performance of all LLMs improves significantly, yielding an average F1 score increase of 13.82%. Notably, larger models now outperform SOTA benchmarks, while smaller models such as Qwen-2.5-Coder-3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection and provides solutions to address the challenges in this task.

摘要: WebShell攻击（恶意脚本被注入网络服务器）构成了重大的网络安全威胁。传统的ML和DL方法经常受到需要大量训练数据、灾难性遗忘和较差的概括性等挑战的阻碍。最近，大型语言模型已成为代码相关任务的强大替代方案，但它们在WebShell检测中的潜力仍然没有得到充分的开发。在本文中，我们做出了两项贡献：（1）对七种LLM进行了全面评估，包括GPT-4、LLaMA 3.1 70 B和Qwen 2.5变体，使用26.59 K个PHP脚本的数据集针对传统的基于序列和图形的方法进行基准测试，以及（2）行为功能感知检测（BFAD）框架，旨在解决将LLM应用于该领域的特定挑战。我们的框架集成了三个组件：隔离恶意PHP函数调用的关键函数过滤器、捕获最具行为指示性的代码段的上下文感知代码提取策略，以及通过优先考虑最相关的演示来增强上下文学习的加权行为函数剖析基于区分性功能级配置文件。我们的结果表明，由于其不同的分析策略，较大的LLM可以实现近乎完美的精确度，但召回率较低，而较小的模型则表现出相反的权衡。然而，所有基线模型都落后于之前的SOTA方法。随着BFAD的应用，所有LLM的性能都显着提高，F1平均得分提高了13.82%。值得注意的是，大型型号的性能现在优于SOTA基准，而Qwen-2.5-Coder-3B等小型型号的性能与传统方法相比具有竞争力。这项工作是第一个探索LLM用于WebShell检测的可行性和局限性的工作，并提供了解决方案来应对这项任务中的挑战。



## **49. Membership Inference Attacks on LLM-based Recommender Systems**

对基于LLM的推荐系统的成员推断攻击 cs.IR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18665v1) [paper-pdf](http://arxiv.org/pdf/2508.18665v1)

**Authors**: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen

**Abstract**: Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.

摘要: 基于大型语言模型（LLM）的推荐系统（RecSys）可以灵活地调整推荐系统以适应不同的领域。它利用上下文学习（ICL），即提示来定制推荐功能，其中包括敏感的历史用户特定项目交互，例如，隐性反馈，例如点击的项目或明确的产品评论。此类私人信息可能会受到新型隐私攻击。然而，对于这个重要问题还没有进行任何研究。我们设计了四种成员资格推断攻击（MIA），旨在揭示受害者的历史互动是否被系统提示使用。它们是\r {直接询问、幻觉、相似性和中毒攻击}，每一种都利用了LLM或RecSys的独特功能。我们在三个用于开发ICL-LLM RecSys的LLM和两个著名的RecSys基准数据集上仔细评估了它们。结果证实，LLM RecSys上的MIA威胁是现实的：直接询问和中毒攻击显示出显着较高的攻击优势。我们还分析了影响这些攻击的因素，例如系统提示中的射击次数以及受害者在射击中的位置。



## **50. Large Language Model-Based Framework for Explainable Cyberattack Detection in Automatic Generation Control Systems**

自动发电控制系统中基于大语言模型的可解释网络攻击检测框架 cs.CR

Accepted Paper

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2507.22239v2) [paper-pdf](http://arxiv.org/pdf/2507.22239v2)

**Authors**: Muhammad Sharshar, Ahmad Mohammad Saber, Davor Svetinovic, Amr M. Youssef, Deepa Kundur, Ehab F. El-Saadany

**Abstract**: The increasing digitization of smart grids has improved operational efficiency but also introduced new cybersecurity vulnerabilities, such as False Data Injection Attacks (FDIAs) targeting Automatic Generation Control (AGC) systems. While machine learning (ML) and deep learning (DL) models have shown promise in detecting such attacks, their opaque decision-making limits operator trust and real-world applicability. This paper proposes a hybrid framework that integrates lightweight ML-based attack detection with natural language explanations generated by Large Language Models (LLMs). Classifiers such as LightGBM achieve up to 95.13% attack detection accuracy with only 0.004 s inference latency. Upon detecting a cyberattack, the system invokes LLMs, including GPT-3.5 Turbo, GPT-4 Turbo, and GPT-4o mini, to generate human-readable explanation of the event. Evaluated on 100 test samples, GPT-4o mini with 20-shot prompting achieved 93% accuracy in identifying the attack target, a mean absolute error of 0.075 pu in estimating attack magnitude, and 2.19 seconds mean absolute error (MAE) in estimating attack onset. These results demonstrate that the proposed framework effectively balances real-time detection with interpretable, high-fidelity explanations, addressing a critical need for actionable AI in smart grid cybersecurity.

摘要: 智能电网的日益数字化提高了运营效率，但也引入了新的网络安全漏洞，例如针对自动发电控制（ATC）系统的虚假数据注入攻击（FDIA）。虽然机器学习（ML）和深度学习（DL）模型在检测此类攻击方面表现出了希望，但其不透明的决策限制了操作员的信任和现实世界的适用性。本文提出了一种混合框架，将轻量级基于ML的攻击检测与大型语言模型（LLM）生成的自然语言解释集成在一起。LightGBM等分类器可实现高达95.13%的攻击检测准确率，推理延迟仅为0.004秒。检测到网络攻击后，系统会调用LLM（包括GPT-3.5涡轮、GPT-4涡轮和GPT-4 o mini）来生成人类可读的事件解释。经过100个测试样本的评估，带20发提示的GPT-4 o mini识别攻击目标的准确率达到了93%，估计攻击幅度的平均绝对误差为0.075 pu，估计攻击开始的平均绝对误差为2.19秒。这些结果表明，拟议的框架有效地平衡了实时检测与可解释的高保真解释，满足了智能电网网络安全中对可操作人工智能的迫切需求。



