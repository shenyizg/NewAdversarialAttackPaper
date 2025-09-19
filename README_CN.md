# Latest Adversarial Attack Papers
**update at 2025-09-19 15:29:23**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction**

超越表面对齐：通过概率简化拒绝指示重建LLM安全机制 cs.CR

Accepted by EMNLP2025 Finding

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15202v1) [paper-pdf](http://arxiv.org/pdf/2509.15202v1)

**Authors**: Yuanbo Xie, Yingjie Zhang, Tianyun Liu, Duohe Ma, Tingwen Liu

**Abstract**: Jailbreak attacks pose persistent threats to large language models (LLMs). Current safety alignment methods have attempted to address these issues, but they experience two significant limitations: insufficient safety alignment depth and unrobust internal defense mechanisms. These limitations make them vulnerable to adversarial attacks such as prefilling and refusal direction manipulation. We introduce DeepRefusal, a robust safety alignment framework that overcomes these issues. DeepRefusal forces the model to dynamically rebuild its refusal mechanisms from jailbreak states. This is achieved by probabilistically ablating the refusal direction across layers and token depths during fine-tuning. Our method not only defends against prefilling and refusal direction attacks but also demonstrates strong resilience against other unseen jailbreak strategies. Extensive evaluations on four open-source LLM families and six representative attacks show that DeepRefusal reduces attack success rates by approximately 95%, while maintaining model capabilities with minimal performance degradation.

摘要: 越狱攻击对大型语言模型（LLM）构成持续威胁。当前的安全对齐方法试图解决这些问题，但它们遇到了两个重大局限性：安全对齐深度不足和内部防御机制不健全。这些限制使它们容易受到预填充和拒绝方向操纵等敌对攻击。我们引入DeepRefusal，这是一个强大的安全调整框架，可以克服这些问题。DeepRefusal迫使该模型动态重建其来自越狱国家的拒绝机制。这是通过在微调期间概率消除跨层和代币深度的拒绝方向来实现的。我们的方法不仅可以抵御预填充和拒绝方向攻击，而且还表现出对其他看不见的越狱策略的强大韧性。对四个开源LLM系列和六种代表性攻击的广泛评估表明，DeepRefusal将攻击成功率降低了约95%，同时以最小的性能下降保持模型能力。



## **2. AIP: Subverting Retrieval-Augmented Generation via Adversarial Instructional Prompt**

AIP：通过对抗性指令提示颠覆检索增强生成 cs.CV

Accepted at EMNLP 2025 Conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15159v1) [paper-pdf](http://arxiv.org/pdf/2509.15159v1)

**Authors**: Saket S. Chaturvedi, Gaurav Bagwe, Lan Zhang, Xiaoyong Yuan

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by retrieving relevant documents from external sources to improve factual accuracy and verifiability. However, this reliance introduces new attack surfaces within the retrieval pipeline, beyond the LLM itself. While prior RAG attacks have exposed such vulnerabilities, they largely rely on manipulating user queries, which is often infeasible in practice due to fixed or protected user inputs. This narrow focus overlooks a more realistic and stealthy vector: instructional prompts, which are widely reused, publicly shared, and rarely audited. Their implicit trust makes them a compelling target for adversaries to manipulate RAG behavior covertly.   We introduce a novel attack for Adversarial Instructional Prompt (AIP) that exploits adversarial instructional prompts to manipulate RAG outputs by subtly altering retrieval behavior. By shifting the attack surface to the instructional prompts, AIP reveals how trusted yet seemingly benign interface components can be weaponized to degrade system integrity. The attack is crafted to achieve three goals: (1) naturalness, to evade user detection; (2) utility, to encourage use of prompts; and (3) robustness, to remain effective across diverse query variations. We propose a diverse query generation strategy that simulates realistic linguistic variation in user queries, enabling the discovery of prompts that generalize across paraphrases and rephrasings. Building on this, a genetic algorithm-based joint optimization is developed to evolve adversarial prompts by balancing attack success, clean-task utility, and stealthiness. Experimental results show that AIP achieves up to 95.23% ASR while preserving benign functionality. These findings uncover a critical and previously overlooked vulnerability in RAG systems, emphasizing the need to reassess the shared instructional prompts.

摘要: 检索增强生成（RAG）通过从外部源检索相关文档来增强大型语言模型（LLM），以提高事实准确性和可验证性。然而，这种依赖在LLM本身之外的检索管道中引入了新的攻击表面。虽然之前的RAG攻击已经暴露了此类漏洞，但它们在很大程度上依赖于操纵用户查询，而由于用户输入固定或受保护，这在实践中通常是不可行的。这种狭隘的焦点忽视了一个更现实、更隐蔽的载体：教学提示，它们被广泛重复使用、公开共享，而且很少审计。他们的隐性信任使他们成为对手秘密操纵RAG行为的引人注目的目标。   我们引入了一种针对对抗性教学提示（AIP）的新型攻击，该攻击利用对抗性教学提示通过微妙地改变检索行为来操纵RAG输出。通过将攻击面转移到指令提示，AIP揭示了如何将可信但看似良性的接口组件武器化以降低系统完整性。该攻击旨在实现三个目标：（1）自然性，以逃避用户检测;（2）实用性，以鼓励使用提示;（3）稳健性，以在不同的查询变体中保持有效。我们提出了一种多样化的查询生成策略，该策略模拟用户查询中现实的语言变化，从而能够发现在重述和改写中进行概括的提示。在此基础上，开发了基于遗传算法的联合优化，通过平衡攻击成功、干净任务效用和隐蔽性来进化对抗提示。实验结果表明，AIP在保持良性功能的同时实现了高达95.23%的ASB。这些发现揭示了RAG系统中一个以前被忽视的关键漏洞，强调需要重新评估共享的教学提示。



## **3. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2409.13174v3) [paper-pdf](http://arxiv.org/pdf/2409.13174v3)

**Authors**: Hao Cheng, Erjia Xiao, Yichi Wang, Chengyuan Yu, Mengshu Sun, Qiang Zhang, Yijie Guo, Kaidi Xu, Jize Zhang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.

摘要: 最近，在多模式大型语言模型（MLLM）进步的推动下，人们提出了视觉语言动作模型（VLAM），以在机器人操纵任务的开放词汇场景中实现更好的性能。由于操纵任务涉及与物理世界的直接互动，因此在执行该任务期间确保稳健性和安全性始终是一个非常关键的问题。本文通过综合当前对MLLM的安全性研究以及物理世界中操纵任务的具体应用场景，对VLAM在潜在物理威胁面前进行了全面评估。具体来说，我们提出了物理脆弱性评估管道（PVEP），它可以整合尽可能多的视觉模式物理威胁，以评估VLAM的物理稳健性。PVEP中的物理威胁具体包括分发外、基于印刷术的视觉提示和对抗性补丁攻击。通过比较VLAM受到攻击前后的性能波动，我们提供了VLAM如何响应不同物理威胁的可概括的\textBF{\textit{Analyses}。



## **4. GASLITEing the Retrieval: Exploring Vulnerabilities in Dense Embedding-based Search**

GASLITEING检索：探索基于密集嵌入的搜索中的漏洞 cs.CR

Accepted at ACM CCS 2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2412.20953v2) [paper-pdf](http://arxiv.org/pdf/2412.20953v2)

**Authors**: Matan Ben-Tov, Mahmood Sharif

**Abstract**: Dense embedding-based text retrieval$\unicode{x2013}$retrieval of relevant passages from corpora via deep learning encodings$\unicode{x2013}$has emerged as a powerful method attaining state-of-the-art search results and popularizing Retrieval Augmented Generation (RAG). Still, like other search methods, embedding-based retrieval may be susceptible to search-engine optimization (SEO) attacks, where adversaries promote malicious content by introducing adversarial passages to corpora. Prior work has shown such SEO is feasible, mostly demonstrating attacks against retrieval-integrated systems (e.g., RAG). Yet, these consider relaxed SEO threat models (e.g., targeting single queries), use baseline attack methods, and provide small-scale retrieval evaluation, thus obscuring our comprehensive understanding of retrievers' worst-case behavior. This work aims to faithfully and thoroughly assess retrievers' robustness, paving a path to uncover factors related to their susceptibility to SEO. To this end, we, first, propose the GASLITE attack for generating adversarial passages, that$\unicode{x2013}$without relying on the corpus content or modifying the model$\unicode{x2013}$carry adversary-chosen information while achieving high retrieval ranking, consistently outperforming prior approaches. Second, using GASLITE, we extensively evaluate retrievers' robustness, testing nine advanced models under varied threat models, while focusing on pertinent adversaries targeting queries on a specific concept (e.g., a public figure). Amongst our findings: retrievers are highly vulnerable to SEO against concept-specific queries, even under negligible poisoning rates (e.g., $\geq$0.0001% of the corpus), while generalizing across different corpora and query distributions; single-query SEO is completely solved by GASLITE; adaptive attacks demonstrate bypassing common defenses; [...]

摘要: 基于密集嵌入的文本检索$\unicode {x2013}$通过深度学习编码从素材库中检索相关段落$\unicode{x2013}$已成为一种强大的方法，可以获得最先进的搜索结果并普及检索增强生成（RAG）。尽管如此，与其他搜索方法一样，基于嵌入的检索可能容易受到搜索引擎优化（SEO）攻击，对手通过在数据库中引入对抗性段落来宣传恶意内容。之前的工作已经表明这种SEO是可行的，主要展示了针对检索集成系统的攻击（例如，RAG）。然而，这些考虑宽松的SEO威胁模型（例如，针对单个查询），使用基线攻击方法，并提供小规模检索评估，从而模糊了我们对检索者最坏情况行为的全面理解。这项工作旨在忠实、彻底地评估猎犬的稳健性，为揭示与它们对SEO敏感性相关的因素铺平道路。为此，我们首先提出了GASLITE攻击来生成对抗性段落，即$\unicode{x2013}$无需依赖于文集内容或修改模型$\unicode{x2013}$携带对抗性选择的信息，同时实现高检索排名，始终优于先前的方法。其次，使用GASLITE，我们广泛地评估了检索器的鲁棒性，在不同的威胁模型下测试了9个高级模型，同时专注于针对特定概念（例如，公众人物）。在我们的研究结果中：检索器非常容易受到针对特定概念查询的SEO的影响，即使在可以忽略不计的中毒率下（例如，$\geq$0.0001%的语料库），同时在不同的语料库和查询分布中推广; GASLITE完全解决了单查询SEO;自适应攻击表明绕过了常见的防御; [...]



## **5. Timestamp Manipulation: Timestamp-based Nakamoto-style Blockchains are Vulnerable**

时间戳操纵：基于时间戳的中本风格区块链很脆弱 cs.CR

25 pages, 6 figures

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2505.05328v4) [paper-pdf](http://arxiv.org/pdf/2505.05328v4)

**Authors**: Junjie Hu, Sisi Duan

**Abstract**: Nakamoto consensus are the most widely adopted decentralized consensus mechanism in cryptocurrency systems. Since it was proposed in 2008, many studies have focused on analyzing its security. Most of them focus on maximizing the profit of the adversary. Examples include the selfish mining attack [FC '14] and the recent riskless uncle maker (RUM) attack [CCS '23]. In this work, we introduce the Staircase-Unrestricted Uncle Maker (SUUM), the first block withholding attack targeting the timestamp-based Nakamoto-style blockchain. Through block withholding, timestamp manipulation, and difficulty risk control, SUUM adversaries are capable of launching persistent attacks with zero cost and minimal difficulty risk characteristics, indefinitely exploiting rewards from honest participants. This creates a self-reinforcing cycle that threatens the security of blockchains. We conduct a comprehensive and systematic evaluation of SUUM, including the attack conditions, its impact on blockchains, and the difficulty risks. Finally, we further discuss four feasible mitigation measures against SUUM.

摘要: Nakamoto共识是加密货币系统中最广泛采用的去中心化共识机制。自2008年提出以来，许多研究都集中在分析其安全性上。他们中的大多数都专注于使对手的利润最大化。例子包括自私的采矿攻击[FC ' 14]和最近的无风险叔叔制造商（RUM）攻击[CS ' 23]。在这项工作中，我们介绍了Staircase-Unrestricted Uncle Maker（SUUM），这是针对基于时间戳的Nakamoto风格区块链的第一个阻止攻击。通过区块扣留、时间戳操纵和难度风险控制，SUUM攻击者能够以零成本和最小难度风险特征发起持续攻击，无限期地利用诚实参与者的奖励。这创造了一个自我强化的循环，威胁到区块链的安全。我们对SUUM进行了全面系统的评估，包括攻击条件，对区块链的影响以及难度风险。最后，我们进一步讨论了针对SUUM的四项可行的缓解措施。



## **6. Discrete optimal transport is a strong audio adversarial attack**

离散最佳传输是一种强烈的音频对抗攻击 eess.AS

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14959v1) [paper-pdf](http://arxiv.org/pdf/2509.14959v1)

**Authors**: Anton Selitskiy, Akib Shahriyar, Jishnuraj Prakasan

**Abstract**: In this paper, we show that discrete optimal transport (DOT) is an effective black-box adversarial attack against modern audio anti-spoofing countermeasures (CMs). Our attack operates as a post-processing, distribution-alignment step: frame-level WavLM embeddings of generated speech are aligned to an unpaired bona fide pool via entropic OT and a top-$k$ barycentric projection, then decoded with a neural vocoder. Evaluated on ASVspoof2019 and ASVspoof5 with AASIST baselines, DOT yields consistently high equal error rate (EER) across datasets and remains competitive after CM fine-tuning, outperforming several conventional attacks in cross-dataset transfer. Ablation analysis highlights the practical impact of vocoder overlap. Results indicate that distribution-level alignment is a powerful and stable attack surface for deployed CMs.

摘要: 在本文中，我们证明了离散最优传输（DOT）是针对现代音频反欺骗对策（CM）的一种有效的黑匣子对抗攻击。我们的攻击作为后处理、分布对齐步骤进行操作：生成语音的帧级WavLM嵌入通过熵OT和顶级$k$重心投影与未配对的真实池对齐，然后使用神经声码器解码。在ASVspoof 2019和ASVspoof 5上采用AASIST基线进行评估，DOT在数据集之间产生了一致的高等错误率（EER），并且在CM微调后仍然具有竞争力，在跨数据集传输中优于几种传统攻击。消融分析强调了声码器重叠的实际影响。结果表明，对于已部署的CM来说，分布级对齐是一个强大且稳定的攻击表面。



## **7. Cost-Performance Analysis: A Comparative Study of CPU-Based Serverless and GPU-Based Training Architectures**

成本-性能分析：基于MCU的无服务器培训架构和基于GOP的培训架构的比较研究 cs.DC

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14920v1) [paper-pdf](http://arxiv.org/pdf/2509.14920v1)

**Authors**: Amine Barrak, Fabio Petrillo, Fehmi Jaafar

**Abstract**: The field of distributed machine learning (ML) faces increasing demands for scalable and cost-effective training solutions, particularly in the context of large, complex models. Serverless computing has emerged as a promising paradigm to address these challenges by offering dynamic scalability and resource-efficient execution. Building upon our previous work, which introduced the Serverless Peer Integrated for Robust Training (SPIRT) architecture, this paper presents a comparative analysis of several serverless distributed ML architectures. We examine SPIRT alongside established architectures like ScatterReduce, AllReduce, and MLLess, focusing on key metrics such as training time efficiency, cost-effectiveness, communication overhead, and fault tolerance capabilities. Our findings reveal that SPIRT provides significant improvements in reducing training times and communication overhead through strategies such as parallel batch processing and in-database operations facilitated by RedisAI. However, traditional architectures exhibit scalability challenges and varying degrees of vulnerability to faults and adversarial attacks. The cost analysis underscores the long-term economic benefits of SPIRT despite its higher initial setup costs. This study not only highlights the strengths and limitations of current serverless ML architectures but also sets the stage for future research aimed at developing new models that combine the most effective features of existing systems.

摘要: 分布式机器学习（ML）领域面临着对可扩展且具有成本效益的训练解决方案日益增长的需求，特别是在大型复杂模型的背景下。无服务器计算已成为一种有希望的范式，可以通过提供动态可扩展性和资源高效的执行来解决这些挑战。在我们之前的工作（介绍了无服务器对等体集成用于稳健训练（SPERT）架构）的基础上，本文对几种无服务器分布式ML架构进行了比较分析。我们将SPRTI与ScatterReduce、AllReduce和MLLess等已建立的架构一起进行研究，重点关注训练时间效率、成本效益、通信负担和故障容忍能力等关键指标。我们的研究结果表明，SPRTI通过RedisAI促进的并行批处理和数据库内操作等策略，在减少训练时间和通信费用方面提供了显着改进。然而，传统架构面临可扩展性挑战，并且对故障和对抗攻击存在不同程度的脆弱性。尽管初始设置成本较高，但成本分析强调了SPERT的长期经济效益。这项研究不仅强调了当前无服务器ML架构的优势和局限性，而且还为未来的研究奠定了基础，旨在开发结合现有系统最有效功能的新模型。



## **8. Birds look like cars: Adversarial analysis of intrinsically interpretable deep learning**

鸟看起来像汽车：本质上可解释的深度学习的对抗分析 cs.LG

Accepted by Machine Learning

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2503.08636v2) [paper-pdf](http://arxiv.org/pdf/2503.08636v2)

**Authors**: Hubert Baniecki, Przemyslaw Biecek

**Abstract**: A common belief is that intrinsically interpretable deep learning models ensure a correct, intuitive understanding of their behavior and offer greater robustness against accidental errors or intentional manipulation. However, these beliefs have not been comprehensively verified, and growing evidence casts doubt on them. In this paper, we highlight the risks related to overreliance and susceptibility to adversarial manipulation of these so-called "intrinsically (aka inherently) interpretable" models by design. We introduce two strategies for adversarial analysis with prototype manipulation and backdoor attacks against prototype-based networks, and discuss how concept bottleneck models defend against these attacks. Fooling the model's reasoning by exploiting its use of latent prototypes manifests the inherent uninterpretability of deep neural networks, leading to a false sense of security reinforced by a visual confirmation bias. The reported limitations of part-prototype networks put their trustworthiness and applicability into question, motivating further work on the robustness and alignment of (deep) interpretable models.

摘要: 人们普遍认为，本质上可解释的深度学习模型可以确保对其行为的正确、直观的理解，并针对意外错误或故意操纵提供更大的鲁棒性。然而，这些信念尚未得到全面验证，越来越多的证据对它们提出了质疑。在本文中，我们强调了与过度依赖和对这些所谓的“本质上（又名本质上）可解释”模型的对抗性操纵的敏感性相关的风险。我们介绍了两种针对基于原型的网络进行原型操纵和后门攻击的对抗分析策略，并讨论概念瓶颈模型如何防御这些攻击。通过利用潜在原型的使用来欺骗模型的推理，这体现了深度神经网络固有的不可解释性，从而导致视觉确认偏差强化的错误安全感。据报道，部分原型网络的局限性使其可信度和适用性受到质疑，促使人们进一步研究（深度）可解释模型的稳健性和一致性。



## **9. STEP: Structured Training and Evaluation Platform for benchmarking trajectory prediction models**

Step：结构化培训和评估平台，用于基准轨迹预测模型 cs.LG

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14801v1) [paper-pdf](http://arxiv.org/pdf/2509.14801v1)

**Authors**: Julian F. Schumann, Anna Mészáros, Jens Kober, Arkady Zgonnikov

**Abstract**: While trajectory prediction plays a critical role in enabling safe and effective path-planning in automated vehicles, standardized practices for evaluating such models remain underdeveloped. Recent efforts have aimed to unify dataset formats and model interfaces for easier comparisons, yet existing frameworks often fall short in supporting heterogeneous traffic scenarios, joint prediction models, or user documentation. In this work, we introduce STEP -- a new benchmarking framework that addresses these limitations by providing a unified interface for multiple datasets, enforcing consistent training and evaluation conditions, and supporting a wide range of prediction models. We demonstrate the capabilities of STEP in a number of experiments which reveal 1) the limitations of widely-used testing procedures, 2) the importance of joint modeling of agents for better predictions of interactions, and 3) the vulnerability of current state-of-the-art models against both distribution shifts and targeted attacks by adversarial agents. With STEP, we aim to shift the focus from the ``leaderboard'' approach to deeper insights about model behavior and generalization in complex multi-agent settings.

摘要: 虽然轨迹预测在实现自动驾驶汽车安全有效的路径规划方面发挥着关键作用，但评估此类模型的标准化实践仍然欠发达。最近的努力旨在统一数据集格式和模型接口，以更容易进行比较，但现有框架往往无法支持异类交通场景、联合预测模型或用户文档。在这项工作中，我们引入了Step --一种新的基准测试框架，它通过为多个数据集提供统一的接口、强制执行一致的训练和评估条件以及支持广泛的预测模型来解决这些局限性。我们在许多实验中展示了Step的能力，这些实验揭示了1）广泛使用的测试程序的局限性，2）对代理进行联合建模对于更好地预测相互作用的重要性，以及3）当前最先进的模型在对抗性代理的分布变化和定向攻击方面的脆弱性。通过Step，我们的目标是将重点从“排行榜”方法转移到对复杂多智能体环境中的模型行为和概括的更深入见解。



## **10. Top K Enhanced Reinforcement Learning Attacks on Heterogeneous Graph Node Classification**

对异类图节点分类的前K增强强化学习攻击 cs.LG

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2408.01964v2) [paper-pdf](http://arxiv.org/pdf/2408.01964v2)

**Authors**: Honglin Gao, Xiang Li, Yajuan Sun, Gaoxi Xiao

**Abstract**: Graph Neural Networks (GNNs) have attracted substantial interest due to their exceptional performance on graph-based data. However, their robustness, especially on heterogeneous graphs, remains underexplored, particularly against adversarial attacks. This paper proposes HeteroKRLAttack, a targeted evasion black-box attack method for heterogeneous graphs. By integrating reinforcement learning with a Top-K algorithm to reduce the action space, our method efficiently identifies effective attack strategies to disrupt node classification tasks. We validate the effectiveness of HeteroKRLAttack through experiments on multiple heterogeneous graph datasets, showing significant reductions in classification accuracy compared to baseline methods. An ablation study underscores the critical role of the Top-K algorithm in enhancing attack performance. Our findings highlight potential vulnerabilities in current models and provide guidance for future defense strategies against adversarial attacks on heterogeneous graphs.

摘要: 图神经网络（GNN）因其在基于图的数据上的出色性能而引起了人们的广泛兴趣。然而，它们的鲁棒性（尤其是在异类图上）仍然没有得到充分的研究，特别是针对对抗性攻击。本文提出了HeteroKRLAttack，一种针对异类图的有针对性的规避黑匣子攻击方法。通过将强化学习与Top-K算法集成以缩小动作空间，我们的方法有效地识别有效的攻击策略来扰乱节点分类任务。我们通过对多个异类图数据集的实验验证了HeteroKRLAttack的有效性，表明与基线方法相比，分类准确性显着降低。一项消融研究强调了Top-K算法在提高攻击性能方面的关键作用。我们的研究结果强调了当前模型中的潜在漏洞，并为未来针对异类图形上的对抗攻击的防御策略提供了指导。



## **11. Mini-Batch Robustness Verification of Deep Neural Networks**

深度神经网络的小批量鲁棒性验证 cs.LG

30 pages, 12 figures, conference OOPSLA 2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2508.15454v2) [paper-pdf](http://arxiv.org/pdf/2508.15454v2)

**Authors**: Saar Tzour-Shaday, Dana Drachsler-Cohen

**Abstract**: Neural network image classifiers are ubiquitous in many safety-critical applications. However, they are susceptible to adversarial attacks. To understand their robustness to attacks, many local robustness verifiers have been proposed to analyze $\epsilon$-balls of inputs. Yet, existing verifiers introduce a long analysis time or lose too much precision, making them less effective for a large set of inputs. In this work, we propose a new approach to local robustness: group local robustness verification. The key idea is to leverage the similarity of the network computations of certain $\epsilon$-balls to reduce the overall analysis time. We propose BaVerLy, a sound and complete verifier that boosts the local robustness verification of a set of $\epsilon$-balls by dynamically constructing and verifying mini-batches. BaVerLy adaptively identifies successful mini-batch sizes, accordingly constructs mini-batches of $\epsilon$-balls that have similar network computations, and verifies them jointly. If a mini-batch is verified, all its $\epsilon$-balls are proven robust. Otherwise, one $\epsilon$-ball is suspected as not being robust, guiding the refinement. BaVerLy leverages the analysis results to expedite the analysis of that $\epsilon$-ball as well as the analysis of the mini-batch with the other $\epsilon$-balls. We evaluate BaVerLy on fully connected and convolutional networks for MNIST and CIFAR-10. Results show that BaVerLy scales the common one by one verification by 2.3x on average and up to 4.1x, in which case it reduces the total analysis time from 24 hours to 6 hours.

摘要: 神经网络图像分类器在许多安全关键应用中无处不在。然而，他们很容易受到敌对攻击。为了了解它们对攻击的鲁棒性，人们提出了许多本地鲁棒性验证器来分析$\$-输入球。然而，现有的验证器引入了较长的分析时间或失去了太多的精确度，使它们对大量输入的效率较低。在这项工作中，我们提出了一种新的局部鲁棒性方法：组局部鲁棒性验证。关键想法是利用某些$\$-balls的网络计算的相似性来减少总体分析时间。我们提出了BaVerLy，这是一个可靠且完整的验证器，通过动态构建和验证迷你批处理来增强一组$\$-balls的局部鲁棒性验证。BaVerLy自适应地识别成功的迷你批量大小，相应地构建具有类似网络计算的$\$-balls的迷你批量，并联合验证它们。如果迷你批次得到验证，则其所有$\$-球都被证明是稳健的。否则，一个$\$-ball被怀疑不稳健，从而指导细化。BaVerLy利用分析结果来加速对该$\ð $-ball的分析以及对其他$\ð $-ball的迷你批次的分析。我们在MNIST和CIFAR-10的全连接和卷积网络上评估了BaVerLy。结果显示，BaVerLy将常见的逐个验证平均扩展了2.3倍，最多扩展了4.1倍，在这种情况下，总分析时间从24小时减少到6小时。



## **12. MUSE: MCTS-Driven Red Teaming Framework for Enhanced Multi-Turn Dialogue Safety in Large Language Models**

MUSE：MCTS驱动的红色团队框架，用于增强大型语言模型中的多回合对话安全性 cs.CL

EMNLP 2025 main conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14651v1) [paper-pdf](http://arxiv.org/pdf/2509.14651v1)

**Authors**: Siyu Yan, Long Zeng, Xuecheng Wu, Chengcheng Han, Kongcheng Zhang, Chong Peng, Xuezhi Cao, Xunliang Cai, Chenjuan Guo

**Abstract**: As large language models~(LLMs) become widely adopted, ensuring their alignment with human values is crucial to prevent jailbreaks where adversaries manipulate models to produce harmful content. While most defenses target single-turn attacks, real-world usage often involves multi-turn dialogues, exposing models to attacks that exploit conversational context to bypass safety measures. We introduce MUSE, a comprehensive framework tackling multi-turn jailbreaks from both attack and defense angles. For attacks, we propose MUSE-A, a method that uses frame semantics and heuristic tree search to explore diverse semantic trajectories. For defense, we present MUSE-D, a fine-grained safety alignment approach that intervenes early in dialogues to reduce vulnerabilities. Extensive experiments on various models show that MUSE effectively identifies and mitigates multi-turn vulnerabilities. Code is available at \href{https://github.com/yansiyu02/MUSE}{https://github.com/yansiyu02/MUSE}.

摘要: 随着大型语言模型（LLM）的广泛采用，确保它们与人类价值观保持一致对于防止对手操纵模型产生有害内容的越狱至关重要。虽然大多数防御措施都针对单轮攻击，但现实世界的使用通常涉及多轮对话，从而使模型暴露于利用对话上下文绕过安全措施的攻击中。我们引入了MUSE，这是一个从攻击和防御角度解决多回合越狱的综合框架。对于攻击，我们提出了MUE-A，这是一种使用框架语义和启发式树搜索来探索不同的语义轨迹的方法。对于防御，我们提出了MUE-D，这是一种细粒度的安全调整方法，可以在对话中早期干预以减少漏洞。对各种模型的广泛实验表明，MUSE可以有效识别和缓解多回合漏洞。代码可访问\href{https：//github.com/yansiyu02/MUSE}{https：//github.com/yansiyu02/MUSE}。



## **13. Enterprise AI Must Enforce Participant-Aware Access Control**

企业人工智能必须强制执行用户感知访问控制 cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14608v1) [paper-pdf](http://arxiv.org/pdf/2509.14608v1)

**Authors**: Shashank Shreedhar Bhatt, Tanmay Rajore, Khushboo Aggarwal, Ganesh Ananthanarayanan, Ranveer Chandra, Nishanth Chandran, Suyash Choudhury, Divya Gupta, Emre Kiciman, Sumit Kumar Pandey, Srinath Setty, Rahul Sharma, Teijia Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in enterprise settings where they interact with multiple users and are trained or fine-tuned on sensitive internal data. While fine-tuning enhances performance by internalizing domain knowledge, it also introduces a critical security risk: leakage of confidential training data to unauthorized users. These risks are exacerbated when LLMs are combined with Retrieval-Augmented Generation (RAG) pipelines that dynamically fetch contextual documents at inference time.   We demonstrate data exfiltration attacks on AI assistants where adversaries can exploit current fine-tuning and RAG architectures to leak sensitive information by leveraging the lack of access control enforcement. We show that existing defenses, including prompt sanitization, output filtering, system isolation, and training-level privacy mechanisms, are fundamentally probabilistic and fail to offer robust protection against such attacks.   We take the position that only a deterministic and rigorous enforcement of fine-grained access control during both fine-tuning and RAG-based inference can reliably prevent the leakage of sensitive data to unauthorized recipients.   We introduce a framework centered on the principle that any content used in training, retrieval, or generation by an LLM is explicitly authorized for \emph{all users involved in the interaction}. Our approach offers a simple yet powerful paradigm shift for building secure multi-user LLM systems that are grounded in classical access control but adapted to the unique challenges of modern AI workflows. Our solution has been deployed in Microsoft Copilot Tuning, a product offering that enables organizations to fine-tune models using their own enterprise-specific data.

摘要: 大型语言模型（LLM）越来越多地部署在企业环境中，它们与多个用户交互，并根据敏感的内部数据接受培训或微调。虽然微调通过内化领域知识来提高性能，但它也会带来严重的安全风险：机密培训数据泄露给未经授权的用户。当LLM与在推理时动态获取上下文文档的检索增强生成（RAG）管道相结合时，这些风险就会加剧。   我们展示了对人工智能助手的数据泄露攻击，其中对手可以利用当前的微调和RAG架构，通过利用访问控制强制执行的缺乏来泄露敏感信息。我们表明，现有的防御措施，包括即时清理、输出过滤、系统隔离和训练级隐私机制，从根本上来说是概率性的，无法提供针对此类攻击的强有力保护。   我们的立场是，只有在微调和基于RAG的推理期间确定性且严格地执行细粒度的访问控制，才能可靠地防止敏感数据泄露给未经授权的接收者。   我们引入了一个框架，其核心原则是LLM在训练、检索或生成中使用的任何内容都被明确授权给\{参与交互的所有用户}。我们的方法为构建安全的多用户LLM系统提供了简单而强大的范式转变，该系统基于经典的访问控制，但适应现代人工智能工作流程的独特挑战。我们的解决方案已部署在Microsoft Copilot Tuning中，这是一种产品，使组织能够使用自己的企业特定数据微调模型。



## **14. GRADA: Graph-based Reranking against Adversarial Documents Attack**

GRADA：基于图的重新排名对抗文档攻击 cs.IR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2505.07546v3) [paper-pdf](http://arxiv.org/pdf/2505.07546v3)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.

摘要: 检索增强生成（RAG）框架通过集成来自检索文档的外部知识来提高大型语言模型（LLM）的准确性，从而克服模型静态内在知识的局限性。然而，这些系统很容易受到对抗性攻击，这些攻击通过引入对抗性但在语义上与查询相似的文档来操纵检索过程。值得注意的是，虽然这些对抗性文档类似于查询，但它们与检索集中的良性文档表现出弱的相似性。因此，我们提出了一个简单而有效的基于图形的对抗性文档攻击重新排名（GRADA）框架，旨在保留检索质量，同时显着降低对手的成功。我们的研究通过在五个LLM上进行的实验来评估我们的方法的有效性：GPT-3.5-Turbo，GPT-4 o，Llama3.1-8b，Llama3.1- 70 b和Qwen2.5- 7 b。我们使用三个数据集来评估性能，来自Natural Questions数据集的结果表明攻击成功率降低了80%，同时保持了最小的准确性损失。



## **15. Measuring Soft Biometric Leakage in Speaker De-Identification Systems**

测量说话人去识别系统中的软生物特征泄漏 cs.SD

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.14469v1) [paper-pdf](http://arxiv.org/pdf/2509.14469v1)

**Authors**: Seungmin Seo, Oleg Aulov, P. Jonathon Phillips

**Abstract**: We use the term re-identification to refer to the process of recovering the original speaker's identity from anonymized speech outputs. Speaker de-identification systems aim to reduce the risk of re-identification, but most evaluations focus only on individual-level measures and overlook broader risks from soft biometric leakage. We introduce the Soft Biometric Leakage Score (SBLS), a unified method that quantifies resistance to zero-shot inference attacks on non-unique traits such as channel type, age range, dialect, sex of the speaker, or speaking style. SBLS integrates three elements: direct attribute inference using pre-trained classifiers, linkage detection via mutual information analysis, and subgroup robustness across intersecting attributes. Applying SBLS with publicly available classifiers, we show that all five evaluated de-identification systems exhibit significant vulnerabilities. Our results indicate that adversaries using only pre-trained models - without access to original speech or system details - can still reliably recover soft biometric information from anonymized output, exposing fundamental weaknesses that standard distributional metrics fail to capture.

摘要: 我们使用“重新识别”一词来指代从匿名语音输出中恢复原始说话者身份的过程。发言人去识别系统旨在降低重新识别的风险，但大多数评估仅关注个人层面的措施，而忽视了软生物识别泄露的更广泛风险。我们引入了软生物识别泄漏评分（SBLS），这是一种统一的方法，可以量化对非独特特征（例如通道类型、年龄范围、方言、说话者性别或说话风格）的零镜头推理攻击的抵抗力。SBLS集成了三个元素：使用预先训练的分类器进行直接属性推断、通过互信息分析进行链接检测以及交叉属性之间的子组鲁棒性。将SBLS与公开可用的分类器应用，我们表明所有五个评估的去识别系统都表现出显着的漏洞。我们的结果表明，仅使用预训练模型（而不访问原始语音或系统细节）的对手仍然可以从匿名输出中可靠地恢复软生物识别信息，从而暴露了标准分布指标无法捕捉的根本弱点。



## **16. Enhanced Rényi Entropy-Based Post-Quantum Key Agreement with Provable Security and Information-Theoretic Guarantees**

增强的基于Rényi Entropy的后量子密钥协议，具有可证明的安全性和信息理论保证 cs.CR

31 pages, 4 tables

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.00104v2) [paper-pdf](http://arxiv.org/pdf/2509.00104v2)

**Authors**: Ruopengyu Xu, Chenglian Liu

**Abstract**: This paper presents an enhanced post-quantum key agreement protocol based on R\'enyi entropy, addressing vulnerabilities in the original construction while preserving information-theoretic security properties. We develop a theoretical framework leveraging entropy-preserving operations and secret-shared verification to achieve provable security against quantum adversaries. Through entropy amplification techniques and quantum-resistant commitments, the protocol establishes $2^{128}$ quantum security guarantees under the quantum random oracle model. Key innovations include a confidentiality-preserving verification mechanism using distributed polynomial commitments, tightened min-entropy bounds with guaranteed non-negativity, and composable security proofs in the quantum universal composability framework. Unlike computational approaches, our method provides information-theoretic security without hardness assumptions while maintaining polynomial complexity. Theoretical analysis demonstrates resilience against known quantum attack vectors, including Grover-accelerated brute force and quantum memory attacks. The protocol achieves parameterization for 128-bit quantum security with efficient $\mathcal{O}(n^{2})$ communication complexity. Extensions to secure multiparty computation and quantum network applications are established, providing a foundation for long-term cryptographic security.

摘要: 本文提出了一种基于R\' enyi（R ' enyi）的增强型后量子密钥协商协议，解决了原始结构中的漏洞，同时保留了信息论安全属性。我们开发了一个理论框架，利用保留信息的操作和秘密共享验证来实现针对量子对手的可证明安全性。通过熵放大技术和量子抵抗承诺，该协议在量子随机预言模型下建立了价值2亿美元的量子安全保证。关键创新包括使用分布式多项承诺的保密性保护验证机制、具有保证非负性的收紧最小熵界限以及量子普适可组合性框架中的可组合安全性证明。与计算方法不同，我们的方法提供了信息理论安全性，无需硬性假设，同时保持了多元复杂性。理论分析证明了对已知量子攻击载体的弹性，包括Grover加速的暴力和量子存储攻击。该协议实现了128位量子安全的参数化，具有高效的$\mathCal{O}（n^{2}）$通信复杂性。建立了安全多方计算和量子网络应用的扩展，为长期加密安全提供了基础。



## **17. Building the Self-Improvement Loop: Error Detection and Correction in Goal-Oriented Semantic Communications**

建立自我改进循环：面向目标的语义通信中的错误检测和纠正 cs.NI

7 pages, 8 figures, this paper has been accepted for publication in  IEEE CSCN 2024

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2411.01544v2) [paper-pdf](http://arxiv.org/pdf/2411.01544v2)

**Authors**: Peizheng Li, Xinyi Lin, Adnan Aijaz

**Abstract**: Error detection and correction are essential for ensuring robust and reliable operation in modern communication systems, particularly in complex transmission environments. However, discussions on these topics have largely been overlooked in semantic communication (SemCom), which focuses on transmitting meaning rather than symbols, leading to significant improvements in communication efficiency. Despite these advantages, semantic errors -- stemming from discrepancies between transmitted and received meanings -- present a major challenge to system reliability. This paper addresses this gap by proposing a comprehensive framework for detecting and correcting semantic errors in SemCom systems. We formally define semantic error, detection, and correction mechanisms, and identify key sources of semantic errors. To address these challenges, we develop a Gaussian process (GP)-based method for latent space monitoring to detect errors, alongside a human-in-the-loop reinforcement learning (HITL-RL) approach to optimize semantic model configurations using user feedback. Experimental results validate the effectiveness of the proposed methods in mitigating semantic errors under various conditions, including adversarial attacks, input feature changes, physical channel variations, and user preference shifts. This work lays the foundation for more reliable and adaptive SemCom systems with robust semantic error management techniques.

摘要: 错误检测和纠正对于确保现代通信系统（特别是在复杂的传输环境中）稳健可靠的操作至关重要。然而，关于这些主题的讨论在语义通信（SemCom）中很大程度上被忽视了，语义通信的重点是传递意义而不是符号，从而极大地提高了通信效率。尽管有这些优势，但由于传输和接收的含义之间的差异而产生的语义错误对系统可靠性构成了重大挑战。本文通过提出一个用于检测和纠正SemCom系统中语义错误的全面框架来解决这一差距。我们正式定义语义错误、检测和纠正机制，并识别语义错误的关键来源。为了应对这些挑战，我们开发了一种基于高斯过程（GP）的潜在空间监控方法来检测错误，并开发了人在环强化学习（HITL-RL）方法来使用用户反馈优化语义模型配置。实验结果验证了所提出的方法在各种条件下减轻语义错误的有效性，包括对抗攻击、输入特征变化、物理通道变化和用户偏好变化。这项工作为具有强大的语义错误管理技术的更可靠和自适应的SemCom系统奠定了基础。



## **18. Evaluating the Defense Potential of Machine Unlearning against Membership Inference Attacks**

评估机器取消学习针对成员推断攻击的防御潜力 cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2508.16150v3) [paper-pdf](http://arxiv.org/pdf/2508.16150v3)

**Authors**: Aristeidis Sidiropoulos, Christos Chrysanthos Nikolaidis, Theodoros Tsiolakis, Nikolaos Pavlidis, Vasilis Perifanis, Pavlos S. Efraimidis

**Abstract**: Membership Inference Attacks (MIAs) pose a significant privacy risk, as they enable adversaries to determine whether a specific data point was included in the training dataset of a model. While Machine Unlearning is primarily designed as a privacy mechanism to efficiently remove private data from a machine learning model without the need for full retraining, its impact on the susceptibility of models to MIA remains an open question. In this study, we systematically assess the vulnerability of models to MIA after applying state-of-art Machine Unlearning algorithms. Our analysis spans four diverse datasets (two from the image domain and two in tabular format), exploring how different unlearning approaches influence the exposure of models to membership inference. The findings highlight that while Machine Unlearning is not inherently a countermeasure against MIA, the unlearning algorithm and data characteristics can significantly affect a model's vulnerability. This work provides essential insights into the interplay between Machine Unlearning and MIAs, offering guidance for the design of privacy-preserving machine learning systems.

摘要: 会员推断攻击（MIA）构成了重大的隐私风险，因为它们使对手能够确定特定数据点是否包含在模型的训练数据集中。虽然Machine Unlearning主要被设计为一种隐私机制，用于有效地从机器学习模型中删除私人数据，而不需要完全重新培训，但它对模型对MIA敏感性的影响仍然是一个悬而未决的问题。在这项研究中，我们在应用最先进的机器取消学习算法后系统评估了模型对MIA的脆弱性。我们的分析跨越了四个不同的数据集（两个来自图像域，两个以表格形式），探索不同的去学习方法如何影响模型对隶属推理的暴露。研究结果强调，虽然机器取消学习本质上并不是针对MIA的对策，但取消学习算法和数据特征可能会显着影响模型的脆弱性。这项工作为机器非学习和MIA之间的相互作用提供了重要见解，为保护隐私的机器学习系统的设计提供了指导。



## **19. CLMTracing: Black-box User-level Watermarking for Code Language Model Tracing**

CLMTracing：用于代码语言模型跟踪的黑匣子用户级水印 cs.PL

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.13982v1) [paper-pdf](http://arxiv.org/pdf/2509.13982v1)

**Authors**: Boyu Zhang, Ping He, Tianyu Du, Xuhong Zhang, Lei Yun, Kingsum Chow, Jianwei Yin

**Abstract**: With the widespread adoption of open-source code language models (code LMs), intellectual property (IP) protection has become an increasingly critical concern. While current watermarking techniques have the potential to identify the code LM to protect its IP, they have limitations when facing the more practical and complex demand, i.e., offering the individual user-level tracing in the black-box setting. This work presents CLMTracing, a black-box code LM watermarking framework employing the rule-based watermarks and utility-preserving injection method for user-level model tracing. CLMTracing further incorporates a parameter selection algorithm sensitive to the robust watermark and adversarial training to enhance the robustness against watermark removal attacks. Comprehensive evaluations demonstrate CLMTracing is effective across multiple state-of-the-art (SOTA) code LMs, showing significant harmless improvements compared to existing SOTA baselines and strong robustness against various removal attacks.

摘要: 随着开源代码语言模型（代码LM）的广泛采用，知识产权（IP）保护已成为一个日益重要的问题。虽然当前的水印技术有潜力识别代码LM以保护其IP，但当面临更实际和复杂的需求时，它们存在局限性，即在黑匣子设置中提供个人用户级跟踪。这项工作提出了CLMTracing，这是一个黑盒代码LM水印框架，采用基于规则的水印和效用保留注入方法进行用户级模型跟踪。CLMTracing进一步结合了对稳健水印敏感的参数选择算法和对抗训练，以增强针对水印去除攻击的稳健性。全面的评估表明，CLMTracing在多种最先进的（SOTA）代码LM中有效，与现有的SOTA基线相比，显示出显着的无害改进以及对各种删除攻击的强大鲁棒性。



## **20. CyberLLMInstruct: A Pseudo-malicious Dataset Revealing Safety-performance Trade-offs in Cyber Security LLM Fine-tuning**

CyberLLMDirecct：揭示网络安全中安全性能权衡的伪恶意数据集LLM微调 cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2503.09334v3) [paper-pdf](http://arxiv.org/pdf/2503.09334v3)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents both opportunities and critical safety risks. We introduce CyberLLMInstruct, a dataset of 54,928 pseudo-malicious instruction-response pairs spanning cyber security tasks including malware analysis, phishing simulations, and zero-day vulnerabilities. Our comprehensive evaluation using seven open-source LLMs reveals a critical trade-off: while fine-tuning improves cyber security task performance (achieving up to 92.50% accuracy on CyberMetric), it severely compromises safety resilience across all tested models and attack vectors (e.g., Llama 3.1 8B's security score against prompt injection drops from 0.95 to 0.15). The dataset incorporates diverse sources including CTF challenges, academic papers, industry reports, and CVE databases to ensure comprehensive coverage of cyber security domains. Our findings highlight the unique challenges of securing LLMs in adversarial domains and establish the critical need for developing fine-tuning methodologies that balance performance gains with safety preservation in security-sensitive domains.

摘要: 将大型语言模型（LLM）集成到网络安全应用程序中既带来了机遇，也带来了严重的安全风险。我们引入CyberLLMCinsert，这是一个由54，928个伪恶意描述-响应对组成的数据集，涵盖网络安全任务，包括恶意软件分析、网络钓鱼模拟和零日漏洞。我们使用七个开源LLM进行的全面评估揭示了一个关键的权衡：虽然微调可以提高网络安全任务性能（在CyberMetric上实现高达92.50%的准确率），但它严重损害了所有测试模型和攻击载体的安全弹性（例如，Lama 3.1 8B对立即注射的安全评分从0.95下降到0.15）。该数据集融合了多种来源，包括CTF挑战、学术论文、行业报告和UTE数据库，以确保网络安全领域的全面覆盖。我们的研究结果强调了在对抗性领域中保护LLM的独特挑战，并确定了开发微调方法的迫切需求，该方法在安全敏感领域中平衡性能收益与安全保护。



## **21. Graph Neural Networks for Next-Generation-IoT: Recent Advances and Open Challenges**

下一代物联网的图形神经网络：最近的进展和开放的挑战 cs.IT

38 pages, 18 figures, and 6 tables. Accepted by the IEEE COMST

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2412.20634v3) [paper-pdf](http://arxiv.org/pdf/2412.20634v3)

**Authors**: Nguyen Xuan Tung, Le Tung Giang, Bui Duc Son, Seon Geun Jeong, Trinh Van Chien, Won Joo Hwang, Lajos Hanzo

**Abstract**: Graph Neural Networks (GNNs) have emerged as a powerful framework for modeling complex interconnected systems, hence making them particularly well-suited to address the growing challenges of next-generation Internet of Things (NG-IoT) networks. Existing studies remain fragmented, and there is a lack of comprehensive guidance on how GNNs can be systematically applied to NG-IoT systems. As NG-IoT systems evolve toward 6G, they incorporate diverse technologies. These advances promise unprecedented connectivity, sensing, and automation but also introduce significant complexity, requiring new approaches for scalable learning, dynamic optimization, and secure, decentralized decision-making. This survey provides a comprehensive and forward-looking exploration of how GNNs can empower NG-IoT environments. We commence by exploring the fundamental paradigms of GNNs and articulating the motivation for their use in NG-IoT networks. Besides, we intrinsically connect GNNs with the family of low-density parity-check codes, modeling the NG-IoT as dynamic constrained graphs. We highlight the distinct roles of node-, edge-, and graph-level tasks in tackling key challenges and demonstrate the GNNs' ability to overcome the limitations of traditional optimization. We examine the application of GNNs across core NG-enabling technologies and their integration with distributed frameworks to support privacy-preservation and distributed intelligence. We then delve into the challenges posed by adversarial attacks, offering insights into defense mechanisms. Lastly, we examine how GNNs can be integrated with emerging technologies. Our findings highlight the transformative potential of GNNs in improving efficiency, scalability, and security. Finally, we summarize the key lessons learned and outline promising future research directions, along with a set of design guidelines tailored for NG-IoT applications.

摘要: 图形神经网络（GNN）已成为对复杂互连系统进行建模的强大框架，因此特别适合应对下一代物联网（NG-IoT）网络日益增长的挑战。现有的研究仍然支离破碎，并且缺乏关于如何系统地将GNN应用于NG-物联网系统的全面指导。随着NG-物联网系统向6 G发展，它们融合了各种技术。这些进步带来了前所未有的连接性、传感和自动化，但也带来了巨大的复杂性，需要新的方法来进行可扩展的学习、动态优化和安全、分散的决策。这项调查对GNN如何支持NG-IoT环境进行了全面且前瞻性的探索。我们首先探索GNN的基本范式，并阐明它们在NG-物联网网络中使用的动机。此外，我们从本质上将GNN与低密度parity check码家族联系起来，将NG-IoT建模为动态约束图。我们强调了节点、边缘和图形级任务在应对关键挑战方面的独特作用，并展示了GNN克服传统优化局限性的能力。我们研究GNN在核心NG使能技术中的应用及其与分布式框架的集成以支持隐私保护和分布式智能。然后，我们深入研究对抗攻击带来的挑战，提供防御机制的见解。最后，我们研究GNN如何与新兴技术集成。我们的研究结果强调了GNN在提高效率、可扩展性和安全性方面的变革潜力。最后，我们总结了所吸取的关键教训，概述了未来有前途的研究方向，以及一套为NG-物联网应用量身定制的设计指南。



## **22. DiffHash: Text-Guided Targeted Attack via Diffusion Models against Deep Hashing Image Retrieval**

迪夫哈希：通过针对深度哈希图像检索的扩散模型进行文本引导定向攻击 cs.IR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.12824v2) [paper-pdf](http://arxiv.org/pdf/2509.12824v2)

**Authors**: Zechao Liu, Zheng Zhou, Xiangkun Chen, Tao Liang, Dapeng Lang

**Abstract**: Deep hashing models have been widely adopted to tackle the challenges of large-scale image retrieval. However, these approaches face serious security risks due to their vulnerability to adversarial examples. Despite the increasing exploration of targeted attacks on deep hashing models, existing approaches still suffer from a lack of multimodal guidance, reliance on labeling information and dependence on pixel-level operations for attacks. To address these limitations, we proposed DiffHash, a novel diffusion-based targeted attack for deep hashing. Unlike traditional pixel-based attacks that directly modify specific pixels and lack multimodal guidance, our approach focuses on optimizing the latent representations of images, guided by text information generated by a Large Language Model (LLM) for the target image. Furthermore, we designed a multi-space hash alignment network to align the high-dimension image space and text space to the low-dimension binary hash space. During reconstruction, we also incorporated text-guided attention mechanisms to refine adversarial examples, ensuring them aligned with the target semantics while maintaining visual plausibility. Extensive experiments have demonstrated that our method outperforms state-of-the-art (SOTA) targeted attack methods, achieving better black-box transferability and offering more excellent stability across datasets.

摘要: 深度哈希模型已被广泛采用来应对大规模图像检索的挑战。然而，由于这些方法容易受到对抗示例的影响，因此面临严重的安全风险。尽管人们越来越多地探索深度哈希模型的有针对性的攻击，但现有方法仍然缺乏多模式指导、依赖标签信息以及依赖像素级操作进行攻击。为了解决这些限制，我们提出了迪夫哈希，这是一种新型的基于扩散的深度哈希定向攻击。与直接修改特定像素且缺乏多模式指导的传统基于像素的攻击不同，我们的方法重点是优化图像的潜在表示，并由目标图像的大型语言模型（LLM）生成的文本信息指导。此外，我们设计了一个多空间哈希对齐网络，将多维图像空间和文本空间与低维二进制哈希空间对齐。在重建过程中，我们还结合了文本引导的注意力机制来完善对抗性示例，确保它们与目标语义保持一致，同时保持视觉可信性。大量实验表明，我们的方法优于最先进的（SOTA）定向攻击方法，实现了更好的黑匣子可转移性，并在数据集之间提供更出色的稳定性。



## **23. Removal Attack and Defense on AI-generated Content Latent-based Watermarking**

对人工智能生成的内容基于潜伏的水印的删除攻击和防御 cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.11745v2) [paper-pdf](http://arxiv.org/pdf/2509.11745v2)

**Authors**: De Zhang Lee, Han Fang, Hanyi Wang, Ee-Chien Chang

**Abstract**: Digital watermarks can be embedded into AI-generated content (AIGC) by initializing the generation process with starting points sampled from a secret distribution. When combined with pseudorandom error-correcting codes, such watermarked outputs can remain indistinguishable from unwatermarked objects, while maintaining robustness under whitenoise. In this paper, we go beyond indistinguishability and investigate security under removal attacks. We demonstrate that indistinguishability alone does not necessarily guarantee resistance to adversarial removal. Specifically, we propose a novel attack that exploits boundary information leaked by the locations of watermarked objects. This attack significantly reduces the distortion required to remove watermarks -- by up to a factor of $15 \times$ compared to a baseline whitenoise attack under certain settings. To mitigate such attacks, we introduce a defense mechanism that applies a secret transformation to hide the boundary, and prove that the secret transformation effectively rendering any attacker's perturbations equivalent to those of a naive whitenoise adversary. Our empirical evaluations, conducted on multiple versions of Stable Diffusion, validate the effectiveness of both the attack and the proposed defense, highlighting the importance of addressing boundary leakage in latent-based watermarking schemes.

摘要: 通过使用从秘密分发中采样的起点初始化生成过程，数字水印可以嵌入到人工智能生成的内容（AIGC）中。当与伪随机错误纠正码结合时，此类带水印的输出可以与未带水印的对象保持不可区分，同时在白噪音下保持鲁棒性。在本文中，我们超越了不可撤销性，并调查了删除攻击下的安全性。我们证明，仅靠不可撤销性并不一定保证对对抗性清除的抵抗。具体来说，我们提出了一种新颖的攻击，该攻击利用带有水印的对象位置泄露的边界信息。与某些设置下的基线白噪音攻击相比，此攻击显着减少了删除水印所需的失真，最多可减少15美元\x $。为了减轻此类攻击，我们引入了一种防御机制，该机制应用秘密转换来隐藏边界，并证明秘密转换有效地使任何攻击者的扰动等效于天真白噪音对手的扰动。我们对多个版本的稳定扩散进行了经验评估，验证了攻击和拟议防御的有效性，强调了解决基于潜伏的水印方案中边界泄漏的重要性。



## **24. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

对抗性文本的人在循环生成：以藏传文字为例 cs.CL

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2412.12478v4) [paper-pdf](http://arxiv.org/pdf/2412.12478v4)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models excel across various NLP tasks but remain highly vulnerable to textual adversarial attacks. While adversarial text generation is crucial for NLP security, explainability, evaluation, and data augmentation, related work remains overwhelmingly English-centric, leaving the problem of constructing high-quality and sustainable adversarial robustness benchmarks for lower-resourced languages both difficult and understudied. First, method customization for lower-resourced languages is complicated due to linguistic differences and limited resources. Second, automated attacks are prone to generating invalid or ambiguous adversarial texts. Last but not least, language models continuously evolve and may be immune to parts of previously generated adversarial texts. To address these challenges, we introduce HITL-GAT, an interactive system based on a general approach to human-in-the-loop generation of adversarial texts. Additionally, we demonstrate the utility of HITL-GAT through a case study on Tibetan script, employing three customized adversarial text generation methods and establishing its first adversarial robustness benchmark, providing a valuable reference for other lower-resourced languages.

摘要: 基于DNN的语言模型在各种NLP任务中表现出色，但仍然非常容易受到文本对抗攻击。虽然对抗性文本生成对于NLP安全性、可解释性、评估和数据增强至关重要，但相关工作仍然以英语为中心，因此为低资源语言构建高质量和可持续的对抗性鲁棒性基准的问题既困难又未得到充分研究。首先，由于语言差异和有限的资源，低资源语言的方法定制是复杂的。其次，自动化攻击容易生成无效或模糊的对抗性文本。最后但并非最不重要的是，语言模型不断进化，并且可能不受之前生成的部分对抗性文本的影响。为了应对这些挑战，我们引入了HITL-GAT，这是一个基于人机交互生成对抗性文本的通用方法的交互系统。此外，我们还通过藏传文字案例研究展示了HITL-GAT的实用性，采用三种定制的对抗性文本生成方法，并建立了其第一个对抗性鲁棒性基准，为其他资源较少的语言提供了宝贵的参考。



## **25. SoK: How Sensor Attacks Disrupt Autonomous Vehicles: An End-to-end Analysis, Challenges, and Missed Threats**

SoK：传感器攻击如何扰乱自动驾驶车辆：端到端分析、挑战和错过的威胁 cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.11120v3) [paper-pdf](http://arxiv.org/pdf/2509.11120v3)

**Authors**: Qingzhao Zhang, Shaocheng Luo, Z. Morley Mao, Miroslav Pajic, Michael K. Reiter

**Abstract**: Autonomous vehicles, including self-driving cars, robotic ground vehicles, and drones, rely on complex sensor pipelines to ensure safe and reliable operation. However, these safety-critical systems remain vulnerable to adversarial sensor attacks that can compromise their performance and mission success. While extensive research has demonstrated various sensor attack techniques, critical gaps remain in understanding their feasibility in real-world, end-to-end systems. This gap largely stems from the lack of a systematic perspective on how sensor errors propagate through interconnected modules in autonomous systems when autonomous vehicles interact with the physical world.   To bridge this gap, we present a comprehensive survey of autonomous vehicle sensor attacks across platforms, sensor modalities, and attack methods. Central to our analysis is the System Error Propagation Graph (SEPG), a structured demonstration tool that illustrates how sensor attacks propagate through system pipelines, exposing the conditions and dependencies that determine attack feasibility. With the aid of SEPG, our study distills seven key findings that highlight the feasibility challenges of sensor attacks and uncovers eleven previously overlooked attack vectors exploiting inter-module interactions, several of which we validate through proof-of-concept experiments. Additionally, we demonstrate how large language models (LLMs) can automate aspects of SEPG construction and cross-validate expert analysis, showcasing the promise of AI-assisted security evaluation.

摘要: 自动驾驶汽车、机器人地面车辆和无人机等自动驾驶车辆依赖复杂的传感器管道来确保安全可靠的运行。然而，这些安全关键系统仍然容易受到对抗性传感器攻击，这可能会损害其性能和任务成功。虽然广泛的研究已经证明了各种传感器攻击技术，但在了解其在现实世界的端到端系统中的可行性方面仍然存在重大差距。这一差距很大程度上源于缺乏对自动驾驶汽车与物理世界互动时传感器误差如何通过自动驾驶系统中的互连模块传播的系统视角。   为了弥合这一差距，我们对跨平台、传感器模式和攻击方法的自动驾驶汽车传感器攻击进行了全面调查。我们分析的核心是系统错误传播图（SEPG），这是一种结构化演示工具，它说明了传感器攻击如何通过系统管道传播，揭示了决定攻击可行性的条件和依赖性。在SEPG的帮助下，我们的研究提炼了七个关键发现，这些发现凸显了传感器攻击的可行性挑战，并揭示了11个以前被忽视的利用模块间交互的攻击载体，其中一些我们通过概念验证实验进行了验证。此外，我们还展示了大型语言模型（LLM）如何自动化SEPG构建的各个方面和交叉验证专家分析，展示了人工智能辅助安全评估的前景。



## **26. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

将逻辑与自身对立：通过对比问题探索模型辩护 cs.CL

Accepted at EMNLP 2025 (Main)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2501.01872v5) [paper-pdf](http://arxiv.org/pdf/2501.01872v5)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.

摘要: 尽管大型语言模型与人类价值观和道德原则广泛一致，但仍然容易受到利用其推理能力的复杂越狱攻击。现有的安全措施通常检测到明显的恶意意图，但无法解决微妙的、推理驱动的漏洞。在这项工作中，我们引入了POATE（极反相查询生成、对抗模板构建和搜索），这是一种新颖的越狱技术，利用对比推理来引发不道德的反应。POATE精心设计了语义上相反的意图，并将它们与对抗模板集成，以非凡的微妙性引导模型走向有害的输出。我们对参数大小不同的六个不同语言模型家族进行了广泛的评估，以证明攻击的稳健性，与现有方法相比，实现了显着更高的攻击成功率（~44%）。为了解决这个问题，我们提出了意图感知CoT和反向思维CoT，它们分解查询以检测恶意意图，并反向推理以评估和拒绝有害响应。这些方法增强了推理的稳健性并加强了模型对对抗性利用的防御。



## **27. AQUA-LLM: Evaluating Accuracy, Quantization, and Adversarial Robustness Trade-offs in LLMs for Cybersecurity Question Answering**

AQUA-LLM：评估LLM中网络安全问题解答的准确性、量化和对抗性鲁棒性权衡 cs.CR

Accepted by the 24th IEEE International Conference on Machine  Learning and Applications (ICMLA'25)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13514v1) [paper-pdf](http://arxiv.org/pdf/2509.13514v1)

**Authors**: Onat Gungor, Roshan Sood, Harold Wang, Tajana Rosing

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong potential for cybersecurity question answering (QA), supporting decision-making in real-time threat detection and response workflows. However, their substantial computational demands pose significant challenges for deployment on resource-constrained edge devices. Quantization, a widely adopted model compression technique, can alleviate these constraints. Nevertheless, quantization may degrade model accuracy and increase susceptibility to adversarial attacks. Fine-tuning offers a potential means to mitigate these limitations, but its effectiveness when combined with quantization remains insufficiently explored. Hence, it is essential to understand the trade-offs among accuracy, efficiency, and robustness. We propose AQUA-LLM, an evaluation framework designed to benchmark several state-of-the-art small LLMs under four distinct configurations: base, quantized-only, fine-tuned, and fine-tuned combined with quantization, specifically for cybersecurity QA. Our results demonstrate that quantization alone yields the lowest accuracy and robustness despite improving efficiency. In contrast, combining quantization with fine-tuning enhances both LLM robustness and predictive performance, achieving an optimal balance of accuracy, robustness, and efficiency. These findings highlight the critical need for quantization-aware, robustness-preserving fine-tuning methodologies to enable the robust and efficient deployment of LLMs for cybersecurity QA.

摘要: 大型语言模型（LLM）最近在网络安全问题回答（QA）方面展示了强大的潜力，支持实时威胁检测和响应工作流程中的决策。然而，它们巨大的计算需求对资源受限的边缘设备上的部署构成了重大挑战。量化是一种广泛采用的模型压缩技术，可以缓解这些限制。然而，量化可能会降低模型准确性并增加对对抗攻击的敏感性。微调提供了一种潜在的手段，以减轻这些限制，但其有效性时，结合量化仍然没有得到充分的探讨。因此，了解准确性、效率和鲁棒性之间的权衡至关重要。我们提出了AQUA-LLM，这是一个评估框架，旨在对四种不同配置下的几种最先进的小型LLM进行基准测试：基础，仅量化，微调和微调与量化相结合，专门用于网络安全QA。我们的研究结果表明，量化单独产生最低的准确性和鲁棒性，尽管提高了效率。相比之下，量化与微调相结合可以增强LLM稳健性和预测性能，实现准确性、稳健性和效率的最佳平衡。这些发现凸显了对量化感知、鲁棒性保持微调方法的迫切需求，以实现网络安全QA的LLM稳健、高效的部署。



## **28. JANUS: A Dual-Constraint Generative Framework for Stealthy Node Injection Attacks**

JANUS：一个用于隐形节点注入攻击的双约束生成框架 cs.LG

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13266v1) [paper-pdf](http://arxiv.org/pdf/2509.13266v1)

**Authors**: Jiahao Zhang, Xiaobing Pei, Zhaokun Zhong, Wenqiang Hao, Zhenghao Tang

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable performance across various applications, yet they are vulnerable to sophisticated adversarial attacks, particularly node injection attacks. The success of such attacks heavily relies on their stealthiness, the ability to blend in with the original graph and evade detection. However, existing methods often achieve stealthiness by relying on indirect proxy metrics, lacking consideration for the fundamental characteristics of the injected content, or focusing only on imitating local structures, which leads to the problem of local myopia. To overcome these limitations, we propose a dual-constraint stealthy node injection framework, called Joint Alignment of Nodal and Universal Structures (JANUS). At the local level, we introduce a local feature manifold alignment strategy to achieve geometric consistency in the feature space. At the global level, we incorporate structured latent variables and maximize the mutual information with the generated structures, ensuring the injected structures are consistent with the semantic patterns of the original graph. We model the injection attack as a sequential decision process, which is optimized by a reinforcement learning agent. Experiments on multiple standard datasets demonstrate that the JANUS framework significantly outperforms existing methods in terms of both attack effectiveness and stealthiness.

摘要: 图形神经网络（GNN）在各种应用程序中表现出了出色的性能，但它们很容易受到复杂的对抗攻击，尤其是节点注入攻击。此类攻击的成功在很大程度上依赖于它们的隐蔽性、融入原始图表并逃避检测的能力。然而，现有的方法往往通过依赖间接代理指标来实现隐蔽性，缺乏对注入内容的基本特征的考虑，或者只专注于模仿局部结构，从而导致局部近视的问题。为了克服这些限制，我们提出了一种双约束隐形节点注入框架，称为节点和通用结构联合对齐（JANUS）。在局部层次上，我们引入了局部特征流形对齐策略，以实现特征空间的几何一致性。在全局层次上，我们将结构化的潜变量和最大化的互信息与生成的结构，确保注入的结构是一致的原始图的语义模式。我们将注入攻击建模为一个顺序决策过程，并通过强化学习代理进行优化。在多个标准数据集上的实验表明，JANUS框架在攻击有效性和隐蔽性方面明显优于现有方法。



## **29. Chernoff Information as a Privacy Constraint for Adversarial Classification and Membership Advantage**

冲突信息作为对抗性分类和成员优势的隐私约束 cs.IT

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2403.10307v3) [paper-pdf](http://arxiv.org/pdf/2403.10307v3)

**Authors**: Ayşe Ünsal, Melek Önen

**Abstract**: This work inspects a privacy metric based on Chernoff information, namely Chernoff differential privacy, due to its significance in characterization of the optimal classifier's performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we characterize the relationship between $\varepsilon\textrm{-}$differential privacy, the best error exponent of one of the errors (when the other is fixed) and the best average error exponent. Accordingly, we re-derive Chernoff differential privacy in connection with $\varepsilon\textrm{-}$differential privacy using the Radon-Nikodym derivative, and prove its relation with Kullback-Leibler (KL) differential privacy. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$ and the impact of the adversary's attack in Laplace mechanisms. Lastly, we introduce a new upper bound on adversary's membership advantage in membership inference attacks using Chernoff DP and numerically compare its performance with existing alternatives based on $(\varepsilon, \delta)\textrm{-}$differential privacy in the literature.

摘要: 这项工作检查了基于Timoff信息的隐私指标，即Timoff差异隐私，因为它在描述最佳分类器性能方面具有重要意义。对抗性分类，就像任何其他分类问题一样，在二元分类的情况下，都是围绕决定任何一个类别时（平均或正确检测）错误概率的最小化而建立的。与经典假设测试问题不同，其中虚警和误判概率分别处理，导致最佳错误指数的不对称行为，在这项工作中，我们描述了$\varepð\texttrm {-}$差异隐私、其中一个错误的最佳错误指数（当另一个错误是固定的时）和最佳平均错误指数之间的关系。因此，我们使用Radon-Nikodym衍生物重新推导与$\varepð\textrm{-}$差异隐私相关的Timoff差异隐私，并证明其与Kullback-Leibler（KL）差异隐私的关系。随后，我们给出了数值评估结果，该结果表明，作为隐私参数$\varepð $和对手攻击在拉普拉斯机制中的影响的函数，Lattoff信息优于Kullback-Leibler分歧。最后，我们在使用Deliverff DP的成员资格推断攻击中引入了对手成员资格优势的新上限，并将其性能与文献中基于$（\varepð，\delta）\texttrm {-}$差异隐私的现有替代方案进行了数字比较。



## **30. Detection of Synthetic Face Images: Accuracy, Robustness, Generalization**

合成人脸图像检测：准确性、鲁棒性、概括性 cs.CV

The paper was presented at the DAGM German Conference on Pattern  Recognition (GCPR), 2025

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2406.17547v2) [paper-pdf](http://arxiv.org/pdf/2406.17547v2)

**Authors**: Nela Petrzelkova, Jan Cech

**Abstract**: An experimental study on detecting synthetic face images is presented. We collected a dataset, called FF5, of five fake face image generators, including recent diffusion models. We find that a simple model trained on a specific image generator can achieve near-perfect accuracy in separating synthetic and real images. The model handles common image distortions (reduced resolution, compression) by using data augmentation. Moreover, partial manipulations, where synthetic images are blended into real ones by inpainting, are identified and the area of the manipulation is localized by a simple model of YOLO architecture. However, the model turned out to be vulnerable to adversarial attacks and does not generalize to unseen generators. Failure to generalize to detect images produced by a newer generator also occurs for recent state-of-the-art methods, which we tested on Realistic Vision, a fine-tuned version of StabilityAI's Stable Diffusion image generator.

摘要: 进行了合成人脸图像检测的实验研究。我们收集了一个名为FF 5的数据集，包含五个假面部图像生成器，包括最近的扩散模型。我们发现，在特定图像生成器上训练的简单模型可以在分离合成图像和真实图像方面实现近乎完美的准确性。该模型通过使用数据增强来处理常见的图像失真（分辨率降低、压缩）。此外，还可以识别部分操纵（通过修补将合成图像混合到真实图像中），并通过YOLO架构的简单模型来本地化操纵区域。然而，事实证明，该模型很容易受到对抗攻击，并且不能推广到看不见的生成器。最近的最先进方法也会出现无法概括检测由较新生成器产生的图像的情况，我们在Realistic Vision上进行了测试，Realistic Vision是StabilityAI的Stable Dispatch图像生成器的微调版本。



## **31. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

大型多模式模型的鲁棒适应用于检索增强仇恨模因检测 cs.CL

EMNLP 2025 Main (Oral)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2502.13061v4) [paper-pdf](http://arxiv.org/pdf/2502.13061v4)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL

摘要: 仇恨模因已成为互联网上的一个重要问题，需要强大的自动化检测系统。虽然大型多模式模型（LSYS）在仇恨模因检测方面表现出了希望，但它们面临着显着的挑战，例如次优的性能和有限的域外概括能力。最近的研究进一步揭示了在这种环境下将监督微调（SFT）和上下文学习应用于LSYS时的局限性。为了解决这些问题，我们提出了一个用于仇恨模因检测的鲁棒适应框架，该框架可以增强领域内准确性和跨领域概括性，同时保留Letts的一般视觉语言能力。分析表明，与SFT模型相比，我们的方法在对抗攻击下实现了更好的鲁棒性。对六个模因分类数据集的实验表明，我们的方法实现了最先进的性能，优于更大的代理系统。此外，与标准SFT相比，我们的方法为解释仇恨内容生成了更高质量的理由，增强了模型的可解释性。代码可访问https://github.com/JingbiaoMei/RGCL



## **32. Bridging Threat Models and Detections: Formal Verification via CADP**

桥梁威胁模型和检测：通过CADP进行正式验证 cs.CR

In Proceedings FROM 2025, arXiv:2509.11877

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13035v1) [paper-pdf](http://arxiv.org/pdf/2509.13035v1)

**Authors**: Dumitru-Bogdan Prelipcean, Cătălin Dima

**Abstract**: Threat detection systems rely on rule-based logic to identify adversarial behaviors, yet the conformance of these rules to high-level threat models is rarely verified formally. We present a formal verification framework that models both detection logic and attack trees as labeled transition systems (LTSs), enabling automated conformance checking via bisimulation and weak trace inclusion. Detection rules specified in the Generic Threat Detection Language (GTDL, a general-purpose detection language we formalize in this work) are assigned a compositional operational semantics, and threat models expressed as attack trees are interpreted as LTSs through a structural trace semantics. Both representations are translated to LNT, a modeling language supported by the CADP toolbox. This common semantic domain enables systematic and automated verification of detection coverage. We evaluate our approach on real-world malware scenarios such as LokiBot and Emotet and provide scalability analysis through parametric synthetic models. Results confirm that our methodology identifies semantic mismatches between threat models and detection rules, supports iterative refinement, and scales to realistic threat landscapes.

摘要: 威胁检测系统依赖基于规则的逻辑来识别对抗行为，但这些规则与高级威胁模型的一致性很少得到正式验证。我们提出了一个正式的验证框架，将检测逻辑和攻击树建模为标记转移系统（LTS），通过互模拟和弱跟踪包含实现自动一致性检查。通用威胁检测语言（GTDL，我们在本文中形式化的通用检测语言）中指定的检测规则被分配了组合操作语义，并通过结构跟踪语义将表示为攻击树的威胁模型解释为LTS。这两种表示都被翻译为LNT，这是CADP工具箱支持的建模语言。这个通用的语义域能够系统化、自动化地验证检测覆盖范围。我们评估我们针对LokiBot和Objetet等现实世界恶意软件场景的方法，并通过参数合成模型提供可扩展性分析。结果证实，我们的方法可以识别威胁模型和检测规则之间的语义不匹配，支持迭代细化，并扩展到现实的威胁格局。



## **33. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

语音认证和反欺骗系统威胁调查 cs.CR

This paper is submitted to the IEEE IoT Journal

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.16843v4) [paper-pdf](http://arxiv.org/pdf/2508.16843v4)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.

摘要: 语音认证发生了重大变化，从依赖手工声学特征的传统系统到可以提取稳健的说话者嵌入的深度学习模型。这一进步扩大了其在金融、智能设备、执法等领域的应用。然而，随着采用率的增加，威胁也随之增加。本调查全面回顾了针对语音认证系统（PAS）和反欺骗对策（CM）的现代威胁格局，包括数据中毒、对抗性、深度伪造和对抗性欺骗攻击。我们按时间顺序追踪语音认证的发展，并研究漏洞如何随着技术进步而演变。对于每种类型的攻击，我们总结了方法论，强调常用的数据集，比较性能和局限性，并使用广泛接受的分类法组织现有文献。通过强调新出现的风险和公开挑战，本调查旨在支持开发更安全、更有弹性的语音认证系统。



## **34. Sy-FAR: Symmetry-based Fair Adversarial Robustness**

Sy-FAR：基于对称性的公平对抗鲁棒性 cs.LG

20 pages, 11 figures

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12939v1) [paper-pdf](http://arxiv.org/pdf/2509.12939v1)

**Authors**: Haneen Najjar, Eyal Ronen, Mahmood Sharif

**Abstract**: Security-critical machine-learning (ML) systems, such as face-recognition systems, are susceptible to adversarial examples, including real-world physically realizable attacks. Various means to boost ML's adversarial robustness have been proposed; however, they typically induce unfair robustness: It is often easier to attack from certain classes or groups than from others. Several techniques have been developed to improve adversarial robustness while seeking perfect fairness between classes. Yet, prior work has focused on settings where security and fairness are less critical. Our insight is that achieving perfect parity in realistic fairness-critical tasks, such as face recognition, is often infeasible -- some classes may be highly similar, leading to more misclassifications between them. Instead, we suggest that seeking symmetry -- i.e., attacks from class $i$ to $j$ would be as successful as from $j$ to $i$ -- is more tractable. Intuitively, symmetry is a desirable because class resemblance is a symmetric relation in most domains. Additionally, as we prove theoretically, symmetry between individuals induces symmetry between any set of sub-groups, in contrast to other fairness notions where group-fairness is often elusive. We develop Sy-FAR, a technique to encourage symmetry while also optimizing adversarial robustness and extensively evaluate it using five datasets, with three model architectures, including against targeted and untargeted realistic attacks. The results show Sy-FAR significantly improves fair adversarial robustness compared to state-of-the-art methods. Moreover, we find that Sy-FAR is faster and more consistent across runs. Notably, Sy-FAR also ameliorates another type of unfairness we discover in this work -- target classes that adversarial examples are likely to be classified into become significantly less vulnerable after inducing symmetry.

摘要: 面部识别系统等对安全至关重要的机器学习（ML）系统容易受到对抗性示例的影响，包括现实世界的物理可实现攻击。人们提出了各种提高ML对抗鲁棒性的方法;然而，它们通常会导致不公平的鲁棒性：从某些类或组进行攻击通常比从其他类或组更容易。人们开发了多种技术来提高对抗鲁棒性，同时寻求类之间的完美公平性。然而，之前的工作重点是安全性和公平性不那么重要的环境。我们的见解是，在现实的公平性关键任务（例如面部识别）中实现完美对等通常是不可行的--某些类别可能高度相似，从而导致它们之间出现更多的错误分类。相反，我们建议寻求对称性--即，从类$i$到$j$的攻击与从$j$到$i$ -的攻击一样成功。直观地说，对称性是可取的，因为类相似性在大多数领域中都是对称关系。此外，正如我们从理论上证明的那样，个体之间的对称性会导致任何一组子群体之间的对称性，这与其他公平性概念相反，在这些公平性概念中，群体公平性往往是难以捉摸的。我们开发了Sy-FAR，这是一种鼓励对称性同时优化对抗鲁棒性的技术，并使用五个数据集，三个模型架构对其进行了广泛的评估，包括针对有针对性和无针对性的现实攻击。结果表明，与最先进的方法相比，Sy-FAR显着提高了公平对抗鲁棒性。此外，我们发现Sy-FAR在运行中更快，更一致。值得注意的是，Sy-FAR还改善了我们在这项工作中发现的另一种类型的不公平性--在诱导对称性后，对抗性示例可能被归类到的目标类别变得明显不那么脆弱。



## **35. Beyond Data Privacy: New Privacy Risks for Large Language Models**

超越数据隐私：大型语言模型的新隐私风险 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.14278v1) [paper-pdf](http://arxiv.org/pdf/2509.14278v1)

**Authors**: Yuntao Du, Zitao Li, Ninghui Li, Bolin Ding

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress in natural language understanding, reasoning, and autonomous decision-making. However, these advancements have also come with significant privacy concerns. While significant research has focused on mitigating the data privacy risks of LLMs during various stages of model training, less attention has been paid to new threats emerging from their deployment. The integration of LLMs into widely used applications and the weaponization of their autonomous abilities have created new privacy vulnerabilities. These vulnerabilities provide opportunities for both inadvertent data leakage and malicious exfiltration from LLM-powered systems. Additionally, adversaries can exploit these systems to launch sophisticated, large-scale privacy attacks, threatening not only individual privacy but also financial security and societal trust. In this paper, we systematically examine these emerging privacy risks of LLMs. We also discuss potential mitigation strategies and call for the research community to broaden its focus beyond data privacy risks, developing new defenses to address the evolving threats posed by increasingly powerful LLMs and LLM-powered systems.

摘要: 大型语言模型（LLM）在自然语言理解、推理和自主决策方面取得了显着进展。然而，这些进步也伴随着严重的隐私问题。虽然大量研究的重点是减轻LLM在模型训练的各个阶段的数据隐私风险，但人们对它们部署中出现的新威胁的关注较少。LLM集成到广泛使用的应用程序中以及其自主能力的武器化产生了新的隐私漏洞。这些漏洞为LLM支持的系统无意中泄露数据和恶意泄露提供了机会。此外，对手可以利用这些系统发起复杂的大规模隐私攻击，不仅威胁个人隐私，还威胁金融安全和社会信任。在本文中，我们系统地研究了LLC这些新出现的隐私风险。我们还讨论了潜在的缓解策略，并呼吁研究界将重点扩大到数据隐私风险之外，开发新的防御措施来应对日益强大的LLM和LLM驱动的系统所构成的不断变化的威胁。



## **36. Adversarial Prompt Distillation for Vision-Language Models**

视觉语言模型的对抗性提示蒸馏 cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.15244v3) [paper-pdf](http://arxiv.org/pdf/2411.15244v3)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.

摘要: 对比图像预训练（CLIP）等大型预训练视觉语言模型（VLM）已被证明容易受到对抗攻击，这引发了人们对其在自动驾驶和医疗诊断等安全关键应用中部署的担忧。对抗性提示调整（APT）是对预训练的VLM进行鲁棒化的一种有希望的方法，它在提示调整的过程中应用对抗性训练。然而，现有的APT方法大多是单模式方法，仅为视觉或文本模式设计提示，从而限制了其稳健性或清晰准确性的有效性。在这项工作中，我们提出了对抗性提示蒸馏（APT），这是一个双峰知识蒸馏框架，通过将APT与多模式知识转移集成来增强APT。APT优化视觉和文本模式的提示，同时从干净的预培训教师CLIP模型中提取知识。对多个基准数据集的广泛实验证明了我们的APT方法在对抗稳健性和精确性方面优于当前最先进的APT方法。APT的有效性也验证了使用非稳健教师来提高微调后的VLM的通用性和稳健性的可能性。



## **37. Gradient-Free Adversarial Purification with Diffusion Models**

采用扩散模型的无干扰对抗净化 cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2501.13336v2) [paper-pdf](http://arxiv.org/pdf/2501.13336v2)

**Authors**: Xuelong Dai, Dong Wang, Xiuzhen Cheng, Bin Xiao

**Abstract**: Adversarial training and adversarial purification are two widely used defense strategies for enhancing model robustness against adversarial attacks. However, adversarial training requires costly retraining, while adversarial purification often suffers from low efficiency. More critically, existing defenses are primarily designed under the perturbation-based adversarial threat model, which is ineffective against recently introduced unrestricted adversarial attacks. In this paper, we propose an effective and efficient defense framework that counters both perturbation-based and unrestricted adversarial attacks. Our approach is motivated by the observation that adversarial examples typically lie near the decision boundary and are highly sensitive to pixel-level perturbations. To address this, we introduce adversarial anti-aliasing, a preprocessing technique that mitigates adversarial noise by reducing the magnitude of pixel-level perturbations. In addition, we propose adversarial super-resolution, which leverages prior knowledge from clean datasets to benignly restore high-quality images from adversarially degraded ones. Unlike image synthesis methods that generate entirely new images, adversarial super-resolution focuses on image restoration, making it more suitable for purification. Importantly, both techniques require no additional training and are computationally efficient since they do not rely on gradient computations. To further improve robustness across diverse datasets, we introduce a contrastive learning-based adversarial deblurring fine-tuning method. By incorporating adversarial priors during fine-tuning on the target dataset, this method enhances purification effectiveness without the need to retrain diffusion models.

摘要: 对抗训练和对抗净化是两种广泛使用的防御策略，用于增强模型针对对抗攻击的稳健性。然而，对抗性训练需要昂贵的再培训，而对抗性净化往往效率低下。更关键的是，现有的防御系统主要是在基于扰动的对抗性威胁模型下设计的，该模型对最近引入的无限制对抗性攻击无效。在本文中，我们提出了一个有效且高效的防御框架，可以对抗基于扰动的和无限制的对抗性攻击。我们的方法的动机是这样一个观察：对抗性示例通常位于决策边界附近，并且对像素级扰动高度敏感。为了解决这个问题，我们引入了对抗性抗锯齿，这是一种预处理技术，通过减少像素级扰动的幅度来减轻对抗性噪音。此外，我们还提出了对抗性超分辨率，它利用来自干净数据集的先验知识，从对抗性退化的图像中良性恢复高质量图像。与生成全新图像的图像合成方法不同，对抗性超分辨率专注于图像恢复，使其更适合净化。重要的是，这两种技术都不需要额外的训练，并且计算效率高，因为它们不依赖于梯度计算。为了进一步提高不同数据集的稳健性，我们引入了一种基于对比学习的对抗去模糊微调方法。通过在对目标数据集进行微调期间纳入对抗先验，该方法增强了净化有效性，而无需重新训练扩散模型。



## **38. Defense-to-Attack: Bypassing Weak Defenses Enables Stronger Jailbreaks in Vision-Language Models**

防御到攻击：击败弱防御，在视觉语言模型中实现更强的越狱 cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12724v1) [paper-pdf](http://arxiv.org/pdf/2509.12724v1)

**Authors**: Yunhan Zhao, Xiang Zheng, Xingjun Ma

**Abstract**: Despite their superb capabilities, Vision-Language Models (VLMs) have been shown to be vulnerable to jailbreak attacks. While recent jailbreaks have achieved notable progress, their effectiveness and efficiency can still be improved. In this work, we reveal an interesting phenomenon: incorporating weak defense into the attack pipeline can significantly enhance both the effectiveness and the efficiency of jailbreaks on VLMs. Building on this insight, we propose Defense2Attack, a novel jailbreak method that bypasses the safety guardrails of VLMs by leveraging defensive patterns to guide jailbreak prompt design. Specifically, Defense2Attack consists of three key components: (1) a visual optimizer that embeds universal adversarial perturbations with affirmative and encouraging semantics; (2) a textual optimizer that refines the input using a defense-styled prompt; and (3) a red-team suffix generator that enhances the jailbreak through reinforcement fine-tuning. We empirically evaluate our method on four VLMs and four safety benchmarks. The results demonstrate that Defense2Attack achieves superior jailbreak performance in a single attempt, outperforming state-of-the-art attack methods that often require multiple tries. Our work offers a new perspective on jailbreaking VLMs.

摘要: 尽管视觉语言模型（VLM）具有出色的功能，但已被证明很容易受到越狱攻击。虽然最近的越狱取得了显着进展，但其有效性和效率仍有待提高。在这项工作中，我们揭示了一个有趣的现象：将弱防御纳入攻击管道中可以显着提高VLM越狱的有效性和效率。基于这一见解，我们提出了Defense 2Attack，这是一种新颖的越狱方法，通过利用防御模式来指导越狱提示设计，绕过了VLM的安全护栏。具体来说，Defense 2Attack由三个关键组件组成：（1）视觉优化器，它嵌入具有肯定和鼓励性语义的通用对抗性扰动;（2）文本优化器，它使用防御风格的提示来细化输入;（3）红队后缀生成器，它通过强化微调来增强越狱。我们根据四个VLM和四个安全基准对我们的方法进行了经验评估。结果表明，Defense 2Attack只需一次尝试即可实现卓越的越狱性能，优于通常需要多次尝试的最先进的攻击方法。我们的工作为越狱VLM提供了新的视角。



## **39. Revisiting Transferable Adversarial Images: Systemization, Evaluation, and New Insights**

重新审视可转移对抗图像：系统化、评估和新见解 cs.CR

TPAMI 2025. Code is available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2310.11850v2) [paper-pdf](http://arxiv.org/pdf/2310.11850v2)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes, Qi Li, Qian Wang, Chao Shen

**Abstract**: Transferable adversarial images raise critical security concerns for computer vision systems in real-world, black-box attack scenarios. Although many transfer attacks have been proposed, existing research lacks a systematic and comprehensive evaluation. In this paper, we systemize transfer attacks into five categories around the general machine learning pipeline and provide the first comprehensive evaluation, with 23 representative attacks against 11 representative defenses, including the recent, transfer-oriented defense and the real-world Google Cloud Vision. In particular, we identify two main problems of existing evaluations: (1) for attack transferability, lack of intra-category analyses with fair hyperparameter settings, and (2) for attack stealthiness, lack of diverse measures. Our evaluation results validate that these problems have indeed caused misleading conclusions and missing points, and addressing them leads to new, \textit{consensus-challenging} insights, such as (1) an early attack, DI, even outperforms all similar follow-up ones, (2) the state-of-the-art (white-box) defense, DiffPure, is even vulnerable to (black-box) transfer attacks, and (3) even under the same $L_p$ constraint, different attacks yield dramatically different stealthiness results regarding diverse imperceptibility metrics, finer-grained measures, and a user study. We hope that our analyses will serve as guidance on properly evaluating transferable adversarial images and advance the design of attacks and defenses. Code is available at https://github.com/ZhengyuZhao/TransferAttackEval.

摘要: 可传输的对抗图像在现实世界的黑匣子攻击场景中给计算机视觉系统带来了关键的安全问题。尽管已经提出了很多转移攻击，但现有的研究缺乏系统、全面的评估。在本文中，我们围绕通用机器学习管道将转移攻击系统化为五类，并提供了首次全面评估，其中包含针对11种代表性防御的23种代表性攻击，包括最近的面向转移的防御和现实世界的Google Cloud Vision。特别是，我们发现了现有评估的两个主要问题：（1）对于攻击可转移性，缺乏具有公平超参数设置的类别内分析，以及（2）对于攻击隐蔽性，缺乏多样化的措施。我们的评估结果证实，这些问题确实导致了误导性结论和缺失点，解决这些问题会带来新的、具有挑战性的见解，例如（1）早期攻击，DI，甚至优于所有类似的后续攻击，（2）最先进的（白盒）防御，DiffPure，甚至容易受到（黑盒）转移攻击，以及（3）即使在相同的$L_p$约束下，不同的攻击产生关于不同的不可感知性度量的显著不同的隐蔽性结果，更细粒度的测量和用户研究。我们希望我们的分析能够指导正确评估可转移的对抗图像并推进攻击和防御的设计。代码可在https://github.com/ZhengyuZhao/TransferAttackEval上获取。



## **40. Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content**

迈向包容性有毒内容适度：解决毒性分类器中对抗性攻击的漏洞 cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12672v1) [paper-pdf](http://arxiv.org/pdf/2509.12672v1)

**Authors**: Shaz Furniturewala, Arkaitz Zubiaga

**Abstract**: The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.

摘要: 由于大型语言模型（LLM）的广泛使用，在线机器生成内容的数量急剧增长，这给内容审核系统带来了新的挑战。传统的内容审核分类器通常根据人类生成的文本进行训练，但由于LLM生成的文本偏离其训练数据以及旨在避免检测的对抗性攻击而遭受错误分类。当今的防御策略是被动的，而不是主动的，因为它们依赖于对抗训练或外部检测模型来识别攻击。在这项工作中，我们的目标是识别毒性分类器中导致错误分类的脆弱组件，提出一种基于机械解释性技术的新型策略。我们的研究重点是微调的BERT和RoBERTa分类器，对跨越各种少数群体的不同数据集进行测试。我们使用对抗攻击技术来识别脆弱的电路。最后，我们抑制了这些脆弱的电路，提高了对抗攻击的性能。我们还提供了对这些脆弱电路的人口统计学层面的见解，揭示了模型训练中的公平性和稳健性差距。我们发现模型具有不同的头部，这些头部要么对性能至关重要，要么容易受到攻击，而抑制脆弱的头部可以提高对抗性输入的性能。我们还发现，不同的头部导致了不同人口群体的脆弱性，这可以为毒性检测模型的更具包容性的开发提供信息。



## **41. CIARD: Cyclic Iterative Adversarial Robustness Distillation**

CIARD：循环迭代对抗稳健蒸馏 cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12633v1) [paper-pdf](http://arxiv.org/pdf/2509.12633v1)

**Authors**: Liming Lu, Shuchao Pang, Xu Zheng, Xiang Gu, Anan Du, Yunhuai Liu, Yongbin Zhou

**Abstract**: Adversarial robustness distillation (ARD) aims to transfer both performance and robustness from teacher model to lightweight student model, enabling resilient performance on resource-constrained scenarios. Though existing ARD approaches enhance student model's robustness, the inevitable by-product leads to the degraded performance on clean examples. We summarize the causes of this problem inherent in existing methods with dual-teacher framework as: 1. The divergent optimization objectives of dual-teacher models, i.e., the clean and robust teachers, impede effective knowledge transfer to the student model, and 2. The iteratively generated adversarial examples during training lead to performance deterioration of the robust teacher model. To address these challenges, we propose a novel Cyclic Iterative ARD (CIARD) method with two key innovations: a. A multi-teacher framework with contrastive push-loss alignment to resolve conflicts in dual-teacher optimization objectives, and b. Continuous adversarial retraining to maintain dynamic teacher robustness against performance degradation from the varying adversarial examples. Extensive experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet demonstrate that CIARD achieves remarkable performance with an average 3.53 improvement in adversarial defense rates across various attack scenarios and a 5.87 increase in clean sample accuracy, establishing a new benchmark for balancing model robustness and generalization. Our code is available at https://github.com/eminentgu/CIARD

摘要: 对抗稳健性蒸馏（ARD）旨在将性能和稳健性从教师模型转移到轻量级学生模型，从而在资源受限的场景中实现弹性性能。尽管现有的ARD方法增强了学生模型的稳健性，但不可避免的副产品导致干净示例的性能下降。我们将现有方法中固有的双教师框架中存在的这个问题的原因总结为：1.双教师模型的不同优化目标，即干净而强大的教师阻碍了知识向学生模式的有效转移，2.训练期间迭代生成的对抗性示例导致稳健教师模型的性能恶化。为了应对这些挑战，我们提出了一种新型的循环迭代ARD（CIARD）方法，具有两个关键创新：a.具有对比推-损失对齐的多教师框架，以解决双教师优化目标中的冲突，以及b。持续的对抗性再培训，以保持动态教师鲁棒性，防止不同对抗性示例带来的绩效下降。CIFAR-10、CIFAR-100和Tiny-ImageNet上的大量实验表明，CIARD实现了非凡的性能，在各种攻击场景中的对抗防御率平均提高了3.53，干净样本准确性提高了5.87，为平衡模型稳健性和概括性建立了新的基准。我们的代码可在https://github.com/eminentgu/CIARD上获取



## **42. Your Compiler is Backdooring Your Model: Understanding and Exploiting Compilation Inconsistency Vulnerabilities in Deep Learning Compilers**

您的编译器正在为您的模型做后门：了解和利用深度学习编译器中的编译不一致漏洞 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11173v2) [paper-pdf](http://arxiv.org/pdf/2509.11173v2)

**Authors**: Simin Chen, Jinjun Peng, Yixin He, Junfeng Yang, Baishakhi Ray

**Abstract**: Deep learning (DL) compilers are core infrastructure in modern DL systems, offering flexibility and scalability beyond vendor-specific libraries. This work uncovers a fundamental vulnerability in their design: can an official, unmodified compiler alter a model's semantics during compilation and introduce hidden backdoors? We study both adversarial and natural settings. In the adversarial case, we craft benign models where triggers have no effect pre-compilation but become effective backdoors after compilation. Tested on six models, three commercial compilers, and two hardware platforms, our attack yields 100% success on triggered inputs while preserving normal accuracy and remaining undetected by state-of-the-art detectors. The attack generalizes across compilers, hardware, and floating-point settings. In the natural setting, we analyze the top 100 HuggingFace models (including one with 220M+ downloads) and find natural triggers in 31 models. This shows that compilers can introduce risks even without adversarial manipulation.   Our results reveal an overlooked threat: unmodified DL compilers can silently alter model semantics. To our knowledge, this is the first work to expose inherent security risks in DL compiler design, opening a new direction for secure and trustworthy ML.

摘要: 深度学习（DL）编译器是现代DL系统的核心基础设施，提供超出供应商特定库的灵活性和可扩展性。这项工作揭示了他们设计中的一个根本漏洞：官方的、未经修改的编译器能否在编译期间改变模型的语义并引入隐藏的后门？我们研究对抗环境和自然环境。在对抗性的情况下，我们构建了良性模型，其中触发器对预编译没有影响，但在编译后成为有效的后门。经过六种型号、三种商业编译器和两个硬件平台的测试，我们的攻击在触发的输入上取得了100%的成功，同时保持正常的准确性并保持未被最先进的检测器检测到。该攻击跨越编译器、硬件和浮点设置进行推广。在自然环境中，我们分析了排名前100的HuggingFace模型（包括下载量超过2.2亿的模型），并在31个模型中找到自然触发因素。这表明，即使没有对抗性操纵，编译器也可能引入风险。   我们的结果揭示了一个被忽视的威胁：未修改的DL编译器可以悄悄改变模型语义。据我们所知，这是第一个暴露DL编译器设计中固有安全风险的工作，为安全和可信的ML开辟了新的方向。



## **43. DisorientLiDAR: Physical Attacks on LiDAR-based Localization**

DisorientLiDART：对基于LiDART的本地化的物理攻击 cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12595v1) [paper-pdf](http://arxiv.org/pdf/2509.12595v1)

**Authors**: Yizhen Lao, Yu Zhang, Ziting Wang, Chengbo Wang, Yifei Xue, Wanpeng Shao

**Abstract**: Deep learning models have been shown to be susceptible to adversarial attacks with visually imperceptible perturbations. Even this poses a serious security challenge for the localization of self-driving cars, there has been very little exploration of attack on it, as most of adversarial attacks have been applied to 3D perception. In this work, we propose a novel adversarial attack framework called DisorientLiDAR targeting LiDAR-based localization. By reverse-engineering localization models (e.g., feature extraction networks), adversaries can identify critical keypoints and strategically remove them, thereby disrupting LiDAR-based localization. Our proposal is first evaluated on three state-of-the-art point-cloud registration models (HRegNet, D3Feat, and GeoTransformer) using the KITTI dataset. Experimental results demonstrate that removing regions containing Top-K keypoints significantly degrades their registration accuracy. We further validate the attack's impact on the Autoware autonomous driving platform, where hiding merely a few critical regions induces noticeable localization drift. Finally, we extended our attacks to the physical world by hiding critical regions with near-infrared absorptive materials, thereby successfully replicate the attack effects observed in KITTI data. This step has been closer toward the realistic physical-world attack that demonstrate the veracity and generality of our proposal.

摘要: 深度学习模型已被证明容易受到具有视觉上难以感知的干扰的对抗攻击。即使这对自动驾驶汽车的本地化构成了严重的安全挑战，但对其攻击的探索却很少，因为大多数对抗性攻击都应用于3D感知。在这项工作中，我们提出了一种名为DisorientLiDART的新型对抗攻击框架，目标是基于LiDART的本地化。通过反向工程本地化模型（例如，特征提取网络），对手可以识别关键关键点并从战略上删除它们，从而破坏基于激光雷达的定位。我们的建议首先使用KITTI数据集在三个最先进的点云配准模型（HRegNet，D3 Feat和GeoTransformer）上进行评估。实验结果表明，删除包含Top-K关键点的区域显着降低其配准精度。我们进一步验证了攻击对Autoware自动驾驶平台的影响，在该平台上，仅隐藏几个关键区域会引起明显的定位漂移。最后，我们通过用近红外吸收材料隐藏关键区域，将攻击扩展到物理世界，从而成功复制了KITTI数据中观察到的攻击效果。这一步更接近现实的物理世界攻击，证明了我们提案的真实性和普遍性。



## **44. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

EmantSleuth：通过语义意图不变性检测提示注入 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.20890v2) [paper-pdf](http://arxiv.org/pdf/2508.20890v2)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.

摘要: 大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，从虚拟助手到自治代理。然而，它们的灵活性也引入了新的攻击向量，特别是提示注入（PI），其中攻击者通过精心制作的输入操纵模型行为。随着攻击者不断地使用释义、混淆甚至多任务注入策略，现有的基准不再足以捕获所有新兴威胁。   为了解决这一差距，我们构建了一个新的基准，系统地扩展了以前的努力。我们的基准涵盖了两种广泛使用的现有基准，同时引入了新的操纵技术和多任务场景，从而提供了更全面的评估设置。我们发现，现有的防御虽然在原始基准上有效，但在我们的基准下表现出明显的弱点，这凸显了对更强大解决方案的需求。我们的关键见解是，虽然攻击形式可能会有所不同，但对手的意图（注入未经授权的任务）保持不变。在这一观察的基础上，我们提出了EmittSleuth，这是一个面向语义的防御框架，它通过对任务级意图而不是表面特征进行推理来检测提示注入。在最先进的基准测试中进行评估后，AktSleuth始终优于现有的防御，同时保持相当的运行时间和成本效率。这些结果表明，基于意图的语义推理提供了一个强大的，有效的，和可推广的策略，以抵御不断发展的即时注入威胁的LLM。



## **45. Exploiting Timing Side-Channels in Quantum Circuits Simulation Via ML-Based Methods**

通过基于ML的方法在量子电路模拟中利用定时边通道 cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12535v1) [paper-pdf](http://arxiv.org/pdf/2509.12535v1)

**Authors**: Ben Dong, Hui Feng, Qian Wang

**Abstract**: As quantum computing advances, quantum circuit simulators serve as critical tools to bridge the current gap caused by limited quantum hardware availability. These simulators are typically deployed on cloud platforms, where users submit proprietary circuit designs for simulation. In this work, we demonstrate a novel timing side-channel attack targeting cloud-based quantum simulators. A co-located malicious process can observe fine-grained execution timing patterns to extract sensitive information about concurrently running quantum circuits. We systematically analyze simulator behavior using the QASMBench benchmark suite, profiling timing and memory characteristics across various circuit executions. Our experimental results show that timing profiles exhibit circuit-dependent patterns that can be effectively classified using pattern recognition techniques, enabling the adversary to infer circuit identities and compromise user confidentiality. We were able to achieve 88% to 99.9% identification rate of quantum circuits based on different datasets. This work highlights previously unexplored security risks in quantum simulation environments and calls for stronger isolation mechanisms to protect user workloads

摘要: 随着量子计算的进步，量子电路模拟器成为弥合当前量子硬件可用性有限造成的差距的关键工具。这些模拟器通常部署在云平台上，用户在云平台上提交专有电路设计进行模拟。在这项工作中，我们展示了一种针对基于云的量子模拟器的新型定时侧通道攻击。位于同一位置的恶意进程可以观察细粒度的执行计时模式，以提取有关并发运行的量子电路的敏感信息。我们使用QASMBench基准套件系统地分析模拟器行为，分析各种电路执行中的计时和内存特征。我们的实验结果表明，时间分布呈现出与电路相关的模式，可以使用模式识别技术有效分类，使对手能够推断电路身份并损害用户的机密性。我们能够基于不同数据集实现88%至99.9%的量子电路识别率。这项工作强调了量子模拟环境中之前未探索的安全风险，并呼吁更强大的隔离机制来保护用户工作负载



## **46. Early Approaches to Adversarial Fine-Tuning for Prompt Injection Defense: A 2022 Study of GPT-3 and Contemporary Models**

即时注射防御对抗微调的早期方法：2022年GPT-3和当代模型的研究 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.14271v1) [paper-pdf](http://arxiv.org/pdf/2509.14271v1)

**Authors**: Gustavo Sandoval, Denys Fenchenko, Junyao Chen

**Abstract**: This paper documents early research conducted in 2022 on defending against prompt injection attacks in large language models, providing historical context for the evolution of this critical security domain. This research focuses on two adversarial attacks against Large Language Models (LLMs): prompt injection and goal hijacking. We examine how to construct these attacks, test them on various LLMs, and compare their effectiveness. We propose and evaluate a novel defense technique called Adversarial Fine-Tuning. Our results show that, without this defense, the attacks succeeded 31\% of the time on GPT-3 series models. When using our Adversarial Fine-Tuning approach, attack success rates were reduced to near zero for smaller GPT-3 variants (Ada, Babbage, Curie), though we note that subsequent research has revealed limitations of fine-tuning-based defenses. We also find that more flexible models exhibit greater vulnerability to these attacks. Consequently, large models such as GPT-3 Davinci are more vulnerable than smaller models like GPT-2. While the specific models tested are now superseded, the core methodology and empirical findings contributed to the foundation of modern prompt injection defense research, including instruction hierarchy systems and constitutional AI approaches.

摘要: 本文记录了2022年针对大型语言模型中的即时注入攻击进行的早期研究，为这一关键安全领域的演变提供了历史背景。本研究重点关注针对大型语言模型（LLM）的两种对抗攻击：提示注入和目标劫持。我们研究如何构建这些攻击，在各种LLM上测试它们，并比较它们的有效性。我们提出并评估了一种名为对抗微调的新型防御技术。我们的结果表明，如果没有这种防御，GPT-3系列模型上的攻击成功率为31%。当使用我们的对抗性微调方法时，较小的GPT-3变体（Ada、Babbage、Curie）的攻击成功率降至接近零，尽管我们注意到后续研究揭示了基于微调的防御的局限性。我们还发现，更灵活的模型对这些攻击表现出更大的脆弱性。因此，GPT-3 Davinci等大型型号比GPT-2等小型型号更容易受到攻击。虽然测试的具体模型现在已被取代，但核心方法论和经验发现为现代即时注射防御研究的基础做出了贡献，包括指令层次系统和宪法人工智能方法。



## **47. How to Beat Nakamoto in the Race**

如何在比赛中击败中本聪 cs.CR

To be presented at the 2025 ACM Conference on Computer and  Communications Security (CCS)

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2508.16202v2) [paper-pdf](http://arxiv.org/pdf/2508.16202v2)

**Authors**: Shu-Jie Cao, Dongning Guo

**Abstract**: This paper studies proof-of-work Nakamoto consensus protocols under bounded network delays, settling two long-standing questions in blockchain security: What is the most effective attack on block safety under a given block confirmation latency? And what is the resulting probability of safety violation? A Markov decision process (MDP) framework is introduced to precisely characterize the system state (including the blocktree and timings of all blocks mined), the adversary's potential actions, and the state transitions due to the adversarial action and the random block arrival processes. An optimal attack, called bait-and-switch, is proposed and proved to maximize the adversary's chance of violating block safety by "beating Nakamoto in the race". The exact probability of this violation is calculated for any given confirmation depth using Markov chain analysis, offering fresh insights into the interplay of network delay, confirmation rules, and blockchain security.

摘要: 本文研究了有限网络延迟下的工作量证明中本聪共识协议，解决了区块链安全中的两个长期存在的问题：在给定的块确认延迟下，对块安全性的最有效攻击是什么？由此产生的安全违规可能性是多少？引入了马尔科夫决策过程（MDP）框架来精确描述系统状态（包括区块树和所有挖掘的块的时间）、对手的潜在行为以及由于对抗行为和随机块到达过程而导致的状态转变。提出并证明了一种称为诱饵和开关的最佳攻击，可以通过“在比赛中击败中本聪”来最大化对手违反区块安全的机会。使用马尔科夫链分析针对任何给定的确认深度计算这种违规的确切概率，为网络延迟、确认规则和区块链安全性的相互作用提供了新的见解。



## **48. Time-Constrained Intelligent Adversaries for Automation Vulnerability Testing: A Multi-Robot Patrol Case Study**

用于自动化漏洞测试的时间约束智能对手：多机器人巡逻案例研究 cs.RO

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11971v1) [paper-pdf](http://arxiv.org/pdf/2509.11971v1)

**Authors**: James C. Ward, Alex Bott, Connor York, Edmund R. Hunt

**Abstract**: Simulating hostile attacks of physical autonomous systems can be a useful tool to examine their robustness to attack and inform vulnerability-aware design. In this work, we examine this through the lens of multi-robot patrol, by presenting a machine learning-based adversary model that observes robot patrol behavior in order to attempt to gain undetected access to a secure environment within a limited time duration. Such a model allows for evaluation of a patrol system against a realistic potential adversary, offering insight into future patrol strategy design. We show that our new model outperforms existing baselines, thus providing a more stringent test, and examine its performance against multiple leading decentralized multi-robot patrol strategies.

摘要: 模拟物理自治系统的敌对攻击可以成为检查其对攻击的稳健性并为可预见性设计提供信息的有用工具。在这项工作中，我们通过多机器人巡逻的视角来检查这一点，提出了一个基于机器学习的对手模型，该模型观察机器人巡逻行为，以尝试在有限的时间内获得对安全环境的未被发现的访问。这样的模型允许针对现实的潜在对手评估巡逻系统，从而深入了解未来的巡逻策略设计。我们表明，我们的新模型优于现有的基线，从而提供了更严格的测试，并针对多种领先的去中心化多机器人巡逻策略检查了其性能。



## **49. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

NeuronStrike：对对齐的LLM的神经元级攻击 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11864v1) [paper-pdf](http://arxiv.org/pdf/2509.11864v1)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.

摘要: 安全一致对于大型语言模型（LLM）的道德部署至关重要，指导它们避免生成有害或不道德的内容。当前的对齐技术，例如有监督的微调和来自人类反馈的强化学习，仍然很脆弱，可以被精心设计的对抗提示绕过。不幸的是，此类攻击依赖于试错，缺乏跨模型的通用性，并且受到可扩展性和可靠性的限制。   本文介绍了NeuroStrike，这是一种新颖且可推广的攻击框架，它利用了对齐技术引入的一个基本漏洞：依赖于负责检测和抑制有害输入的稀疏、专门的安全神经元。我们将NeuroStrike应用于白盒和黑盒设置：在白盒设置中，NeuroStrike通过反馈激活分析识别安全神经元，并在推理期间修剪它们以禁用安全机制。在黑匣子环境中，我们提出了第一次LLM剖析攻击，该攻击通过在开权重代理模型上训练对抗提示生成器，然后将它们部署到黑匣子和专有目标上来利用安全神经元的可移植性。我们对来自主要LLM开发商的20多个开量级LLM进行了评估。通过删除目标层中不到0.6%的神经元，NeuroStrike仅使用普通恶意提示即可实现76.9%的平均攻击成功率（ASB）。此外，Neurostrike将四种多模式LLM推广到对不安全图像输入具有100%的ASB。安全神经元在架构之间有效转移，使11个微调模型的ASB达到78.5%，5个提炼模型的ASB达到77.7%。黑匣子LLM分析攻击在包括Google Gemini系列在内的五种黑匣子型号中实现了63.7%的平均ASB。



## **50. A Practical Adversarial Attack against Sequence-based Deep Learning Malware Classifiers**

针对基于序列的深度学习恶意软件分类器的实用对抗攻击 cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11836v1) [paper-pdf](http://arxiv.org/pdf/2509.11836v1)

**Authors**: Kai Tan, Dongyang Zhan, Lin Ye, Hongli Zhang, Binxing Fang

**Abstract**: Sequence-based deep learning models (e.g., RNNs), can detect malware by analyzing its behavioral sequences. Meanwhile, these models are susceptible to adversarial attacks. Attackers can create adversarial samples that alter the sequence characteristics of behavior sequences to deceive malware classifiers. The existing methods for generating adversarial samples typically involve deleting or replacing crucial behaviors in the original data sequences, or inserting benign behaviors that may violate the behavior constraints. However, these methods that directly manipulate sequences make adversarial samples difficult to implement or apply in practice. In this paper, we propose an adversarial attack approach based on Deep Q-Network and a heuristic backtracking search strategy, which can generate perturbation sequences that satisfy practical conditions for successful attacks. Subsequently, we utilize a novel transformation approach that maps modifications back to the source code, thereby avoiding the need to directly modify the behavior log sequences. We conduct an evaluation of our approach, and the results confirm its effectiveness in generating adversarial samples from real-world malware behavior sequences, which have a high success rate in evading anomaly detection models. Furthermore, our approach is practical and can generate adversarial samples while maintaining the functionality of the modified software.

摘要: 基于序列的深度学习模型（例如，RNN），可以通过分析恶意软件的行为序列来检测恶意软件。与此同时，这些模型很容易受到对抗攻击。攻击者可以创建对抗样本，这些样本改变行为序列的序列特征，以欺骗恶意软件分类器。生成对抗样本的现有方法通常涉及删除或替换原始数据序列中的关键行为，或者插入可能违反行为约束的良性行为。然而，这些直接操纵序列的方法使得对抗性样本难以在实践中实现或应用。本文提出了一种基于Deep Q网络的对抗性攻击方法和启发式回溯搜索策略，该策略可以生成满足成功攻击实际条件的扰动序列。随后，我们利用一种新颖的转换方法，将修改映射回源代码，从而避免了直接修改行为日志序列的需要。我们对我们的方法进行了评估，结果证实了它在从现实世界的恶意软件行为序列生成对抗样本方面的有效性，这些样本在逃避异常检测模型方面具有很高的成功率。此外，我们的方法很实用，可以生成对抗样本，同时保持修改后软件的功能。



