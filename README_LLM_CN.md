# Latest Large Language Model Attack Papers
**update at 2025-12-23 09:35:58**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

有害在线内容的对抗稳健检测：计算设计科学方法 cs.LG

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17367v1) [paper-pdf](https://arxiv.org/pdf/2512.17367v1)

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.

摘要: 社交媒体平台受到仇恨言论、错误信息和极端主义言论等有害内容的困扰。机器学习（ML）模型被广泛采用来检测此类内容;然而，它们仍然极易受到对抗攻击，其中恶意用户会巧妙地修改文本以逃避检测。因此，增强对抗鲁棒性至关重要，需要检测器能够抵御各种攻击（可概括性），同时保持高的总体准确性。然而，同时实现最佳概括性和准确性是一项挑战。遵循计算设计科学范式，本研究采用顺序方法，首先提出了一种新颖的框架（基于大语言模型的样本生成和聚合，LLM-LGA），通过识别文本对抗攻击的关键不变性并利用它们来确保框架内实例化的检测器具有很强的概括性。其次，我们实例化我们的检测器（对抗鲁棒有害在线内容检测器，ARHOCD）具有三个新颖的设计组件来提高检测准确性：（1）利用其互补优势的多个基本检测器的集成;（2）一种新颖的权重分配方法，其基于每个样本的可预测性和每个碱基检测器的能力动态调整权重，权重使用领域知识初始化并通过Bayesian推理更新;以及（3）一种新颖的对抗训练策略，迭代优化基本检测器和权重分配器。我们解决了现有对抗鲁棒性增强研究的几个局限性，并在跨越仇恨言论、谣言和极端主义内容的三个数据集中对ARHOCD进行了实证评估。结果表明，ARHOCD具有很强的概括性，并提高了对抗条件下的检测准确性。



## **2. Cryptanalysis of Pseudorandom Error-Correcting Codes**

伪随机错误纠正码的密码分析 cs.CR

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17310v1) [paper-pdf](https://arxiv.org/pdf/2512.17310v1)

**Authors**: Tianrui Wang, Anyu Wang, Tianshuo Cong, Delong Ran, Jinyuan Liu, Xiaoyun Wang

**Abstract**: Pseudorandom error-correcting codes (PRC) is a novel cryptographic primitive proposed at CRYPTO 2024. Due to the dual capability of pseudorandomness and error correction, PRC has been recognized as a promising foundational component for watermarking AI-generated content. However, the security of PRC has not been thoroughly analyzed, especially with concrete parameters or even in the face of cryptographic attacks. To fill this gap, we present the first cryptanalysis of PRC. We first propose three attacks to challenge the undetectability and robustness assumptions of PRC. Among them, two attacks aim to distinguish PRC-based codewords from plain vectors, and one attack aims to compromise the decoding process of PRC. Our attacks successfully undermine the claimed security guarantees across all parameter configurations. Notably, our attack can detect the presence of a watermark with overwhelming probability at a cost of $2^{22}$ operations. We also validate our approach by attacking real-world large generative models such as DeepSeek and Stable Diffusion. To mitigate our attacks, we further propose three defenses to enhance the security of PRC, including parameter suggestions, implementation suggestions, and constructing a revised key generation algorithm. Our proposed revised key generation function effectively prevents the occurrence of weak keys. However, we highlight that the current PRC-based watermarking scheme still cannot achieve a 128-bit security under our parameter suggestions due to the inherent configurations of large generative models, such as the maximum output length of large language models.

摘要: 伪随机纠错码（PRC）是在2024年国际密码学大会上提出的一种新的密码学原语。由于伪随机性和纠错的双重能力，PRC已被认为是对AI生成的内容进行水印的有前途的基础组件。然而，PRC的安全性还没有得到彻底的分析，特别是在具体的参数，甚至在面对密码攻击。为了填补这一空白，我们提出了PRC的第一个密码分析。我们首先提出三种攻击来挑战PRC的不可检测性和稳健性假设。其中，两次攻击旨在将基于PRC的代码字与纯载体区分开来，一次攻击旨在损害PRC的解码过程。我们的攻击成功地破坏了所有参数配置中声称的安全保证。值得注意的是，我们的攻击可以以压倒性的可能性检测到水印的存在，但操作成本为2美元的^{22}$。我们还通过攻击DeepSeek和Stable Distance等现实世界的大型生成模型来验证我们的方法。为了减轻我们的攻击，我们进一步提出了三种防御措施来增强PRC的安全性，包括参数建议、实施建议和构建修改后的密钥生成算法。我们提出的修改后的密钥生成功能有效地防止了弱密钥的发生。然而，我们强调，由于大型生成模型的固有配置，例如大型语言模型的最大输出长度，当前基于PRC的水印方案在我们的参数建议下仍然无法实现128位安全性。



## **3. Biosecurity-Aware AI: Agentic Risk Auditing of Soft Prompt Attacks on ESM-Based Variant Predictors**

具有生物安全意识的人工智能：对基于ESM的变体预测器的软提示攻击的量化风险审计 cs.CR

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17146v1) [paper-pdf](https://arxiv.org/pdf/2512.17146v1)

**Authors**: Huixin Zhan

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated remarkable success in variant effect prediction. However, their security and robustness under adversarial manipulation remain largely unexplored. To address this gap, we introduce the Secure Agentic Genomic Evaluator (SAGE), an agentic framework for auditing the adversarial vulnerabilities of GFMs. SAGE functions through an interpretable and automated risk auditing loop. It injects soft prompt perturbations, monitors model behavior across training checkpoints, computes risk metrics such as AUROC and AUPR, and generates structured reports with large language model-based narrative explanations. This agentic process enables continuous evaluation of embedding-space robustness without modifying the underlying model. Using SAGE, we find that even state-of-the-art GFMs like ESM2 are sensitive to targeted soft prompt attacks, resulting in measurable performance degradation. These findings reveal critical and previously hidden vulnerabilities in genomic foundation models, showing the importance of agentic risk auditing in securing biomedical applications such as clinical variant interpretation.

摘要: 基因组基础模型（GFM），例如进化规模建模（ESM），在变异效应预测方面取得了显着的成功。然而，它们在对抗操纵下的安全性和稳健性在很大程度上仍未得到探索。为了解决这一差距，我们引入了安全统计基因组评估器（SAGE），这是一个用于审计GFM对抗性漏洞的代理框架。SAGE通过可解释和自动化的风险审计循环发挥作用。它注入软提示扰动，监控训练检查点的模型行为，计算AUROC和AUPR等风险指标，并生成具有基于大型语言模型的叙述性解释的结构化报告。这个代理过程能够连续评估嵌入空间稳健性，而无需修改基础模型。使用SAGE，我们发现即使是ESM 2等最先进的GFM也对有针对性的软提示攻击敏感，从而导致可衡量的性能下降。这些发现揭示了基因组基础模型中关键且先前隐藏的漏洞，表明代理风险审计在确保临床变体解释等生物医学应用方面的重要性。



## **4. From Essence to Defense: Adaptive Semantic-aware Watermarking for Embedding-as-a-Service Copyright Protection**

从本质到防御：用于嵌入即服务版权保护的自适应语义感知水印 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16439v1) [paper-pdf](https://arxiv.org/pdf/2512.16439v1)

**Authors**: Hao Li, Yubing Ren, Yanan Cao, Yingjie Li, Fang Fang, Xuebin Wang

**Abstract**: Benefiting from the superior capabilities of large language models in natural language understanding and generation, Embeddings-as-a-Service (EaaS) has emerged as a successful commercial paradigm on the web platform. However, prior studies have revealed that EaaS is vulnerable to imitation attacks. Existing methods protect the intellectual property of EaaS through watermarking techniques, but they all ignore the most important properties of embedding: semantics, resulting in limited harmlessness and stealthiness. To this end, we propose SemMark, a novel semantic-based watermarking paradigm for EaaS copyright protection. SemMark employs locality-sensitive hashing to partition the semantic space and inject semantic-aware watermarks into specific regions, ensuring that the watermark signals remain imperceptible and diverse. In addition, we introduce the adaptive watermark weight mechanism based on the local outlier factor to preserve the original embedding distribution. Furthermore, we propose Detect-Sampling and Dimensionality-Reduction attacks and construct four scenarios to evaluate the watermarking method. Extensive experiments are conducted on four popular NLP datasets, and SemMark achieves superior verifiability, diversity, stealthiness, and harmlessness.

摘要: 得益于大型语言模型在自然语言理解和生成方面的卓越能力，嵌入式即服务（EaaS）已经成为Web平台上成功的商业模式。然而，之前的研究表明，EaaS容易受到模仿攻击。现有的方法通过水印技术保护EaaS的知识产权，但它们都忽略了嵌入最重要的属性：语义，导致有限的无害性和隐蔽性。为此，我们提出了SemMark，一种新的基于语义的水印范例EaaS版权保护。SemMark采用局部敏感哈希来划分语义空间，并将语义感知水印注入特定区域，确保水印信号保持不可感知和多样性。此外，我们引入了基于局部离群因子的自适应水印权重机制，以保持原始的嵌入分布。此外，我们提出了检测采样和降低分辨率攻击，并构造了四个场景来评估水印方法。在四个流行的NLP数据集上进行了广泛的实验，SemMark实现了卓越的可验证性，多样性，隐蔽性和无害性。



## **5. MemoryGraft: Persistent Compromise of LLM Agents via Poisoned Experience Retrieval**

MemoryGraft：通过中毒经验检索持续损害LLM药物 cs.CR

14 pages, 1 figure, includes appendix

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16962v1) [paper-pdf](https://arxiv.org/pdf/2512.16962v1)

**Authors**: Saksham Sahai Srivastava, Haoyu He

**Abstract**: Large Language Model (LLM) agents increasingly rely on long-term memory and Retrieval-Augmented Generation (RAG) to persist experiences and refine future performance. While this experience learning capability enhances agentic autonomy, it introduces a critical, unexplored attack surface, i.e., the trust boundary between an agent's reasoning core and its own past. In this paper, we introduce MemoryGraft. It is a novel indirect injection attack that compromises agent behavior not through immediate jailbreaks, but by implanting malicious successful experiences into the agent's long-term memory. Unlike traditional prompt injections that are transient, or standard RAG poisoning that targets factual knowledge, MemoryGraft exploits the agent's semantic imitation heuristic which is the tendency to replicate patterns from retrieved successful tasks. We demonstrate that an attacker who can supply benign ingestion-level artifacts that the agent reads during execution can induce it to construct a poisoned RAG store where a small set of malicious procedure templates is persisted alongside benign experiences. When the agent later encounters semantically similar tasks, union retrieval over lexical and embedding similarity reliably surfaces these grafted memories, and the agent adopts the embedded unsafe patterns, leading to persistent behavioral drift across sessions. We validate MemoryGraft on MetaGPT's DataInterpreter agent with GPT-4o and find that a small number of poisoned records can account for a large fraction of retrieved experiences on benign workloads, turning experience-based self-improvement into a vector for stealthy and durable compromise. To facilitate reproducibility and future research, our code and evaluation data are available at https://github.com/Jacobhhy/Agent-Memory-Poisoning.

摘要: 大型语言模型（LLM）代理越来越依赖长期记忆和检索增强生成（RAG）来持久体验并改进未来性能。虽然这种经验学习能力增强了代理人的自主性，但它引入了一个关键的、未探索的攻击表面，即代理人的推理核心与其自己的过去之间的信任边界。在本文中，我们介绍了MemoryGraft。这是一种新颖的间接注入攻击，它不是通过立即越狱，而是通过将恶意的成功体验植入到代理的长期记忆中来损害代理的行为。与传统的短暂提示注入或针对事实知识的标准RAG中毒不同，MemoryGraft利用了代理的语义模仿启发式，即从检索到的成功任务中复制模式的倾向。我们证明，能够提供代理在执行期间读取的良性摄入级工件的攻击者可以诱导其构建一个有毒的RAG存储，其中一小组恶意过程模板与良性体验一起持久存在。当代理后来遇到语义相似的任务时，基于词汇和嵌入相似性的联合检索会可靠地暴露这些嫁接的记忆，并且代理采用嵌入的不安全模式，从而导致跨会话持续的行为漂移。我们在MetaGPT的DataInterpreter代理上使用GPT-4o验证了SecureGraft，发现少量有毒记录可以解释良性工作负载上检索到的体验的大部分，从而将基于经验的自我改进转变为隐形和持久妥协的载体。为了促进重现性和未来的研究，我们的代码和评估数据可在www.example.com上获取。



## **6. In-Context Probing for Membership Inference in Fine-Tuned Language Models**

精调语言模型中成员推理的上下文探索 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16292v1) [paper-pdf](https://arxiv.org/pdf/2512.16292v1)

**Authors**: Zhexi Lu, Hongliang Chi, Nathalie Baracaldo, Swanand Ravindra Kadhe, Yuseok Jeon, Lei Yu

**Abstract**: Membership inference attacks (MIAs) pose a critical privacy threat to fine-tuned large language models (LLMs), especially when models are adapted to domain-specific tasks using sensitive data. While prior black-box MIA techniques rely on confidence scores or token likelihoods, these signals are often entangled with a sample's intrinsic properties - such as content difficulty or rarity - leading to poor generalization and low signal-to-noise ratios. In this paper, we propose ICP-MIA, a novel MIA framework grounded in the theory of training dynamics, particularly the phenomenon of diminishing returns during optimization. We introduce the Optimization Gap as a fundamental signal of membership: at convergence, member samples exhibit minimal remaining loss-reduction potential, while non-members retain significant potential for further optimization. To estimate this gap in a black-box setting, we propose In-Context Probing (ICP), a training-free method that simulates fine-tuning-like behavior via strategically constructed input contexts. We propose two probing strategies: reference-data-based (using semantically similar public samples) and self-perturbation (via masking or generation). Experiments on three tasks and multiple LLMs show that ICP-MIA significantly outperforms prior black-box MIAs, particularly at low false positive rates. We further analyze how reference data alignment, model type, PEFT configurations, and training schedules affect attack effectiveness. Our findings establish ICP-MIA as a practical and theoretically grounded framework for auditing privacy risks in deployed LLMs.

摘要: 成员资格推理攻击（MIA）对微调的大型语言模型（LLM）构成严重的隐私威胁，尤其是当模型使用敏感数据适应特定领域任务时。虽然先前的黑匣子MIA技术依赖于置信度分数或代币可能性，但这些信号通常与样本的内在属性（例如内容难度或稀有性）纠缠在一起，导致概括性较差和低信噪比。在本文中，我们提出了ICP-MIA，这是一种基于训练动力学理论的新型MIA框架，特别是优化过程中的回报递减现象。我们引入优化差距作为成员资格的基本信号：在收敛时，成员样本表现出最小的剩余损失减少潜力，而非成员保留了进一步优化的显着潜力。为了估计黑匣子环境中的这一差距，我们提出了In-Context Probing（ICP），这是一种免训练的方法，通过策略性构建的输入上下文模拟类似微调的行为。我们提出了两种探测策略：基于参考数据（使用语义相似的公共样本）和自我扰动（通过掩蔽或生成）。三项任务和多个LLM的实验表明，ICP-MIA显着优于之前的黑匣子MIA，尤其是在低假阳性率下。我们进一步分析参考数据对齐、模型类型、PEFT配置和训练计划如何影响攻击有效性。我们的研究结果将ICP-MIA确立为审计已部署的LLM隐私风险的实用且理论基础的框架。



## **7. DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack**

DualGuard：双数据流大型语言模型水印防御，防止重述和欺骗攻击 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16182v1) [paper-pdf](https://arxiv.org/pdf/2512.16182v1)

**Authors**: Hao Li, Yubing Ren, Yanan Cao, Yingjie Li, Fang Fang, Shi Wang, Li Guo

**Abstract**: With the rapid development of cloud-based services, large language models (LLMs) have become increasingly accessible through various web platforms. However, this accessibility has also led to growing risks of model abuse. LLM watermarking has emerged as an effective approach to mitigate such misuse and protect intellectual property. Existing watermarking algorithms, however, primarily focus on defending against paraphrase attacks while overlooking piggyback spoofing attacks, which can inject harmful content, compromise watermark reliability, and undermine trust in attribution. To address this limitation, we propose DualGuard, the first watermarking algorithm capable of defending against both paraphrase and spoofing attacks. DualGuard employs the adaptive dual-stream watermarking mechanism, in which two complementary watermark signals are dynamically injected based on the semantic content. This design enables DualGuard not only to detect but also to trace spoofing attacks, thereby ensuring reliable and trustworthy watermark detection. Extensive experiments conducted across multiple datasets and language models demonstrate that DualGuard achieves excellent detectability, robustness, traceability, and text quality, effectively advancing the state of LLM watermarking for real-world applications.

摘要: 随着基于云的服务的快速发展，大型语言模型（LLM）越来越多地可以通过各种网络平台访问。然而，这种可及性也导致了模型滥用的风险越来越大。LLM水印已成为减少此类滥用和保护知识产权的有效方法。然而，现有的水印算法主要专注于防御重述攻击，同时忽略了背负欺骗攻击，这种攻击可能会注入有害内容、损害水印可靠性并破坏对归因的信任。为了解决这一局限性，我们提出了DualGuard，这是第一个能够防御重述和欺骗攻击的水印算法。DualGuard采用自适应双流水印机制，根据语义内容动态注入两个互补的水印信号。该设计使DualGuard不仅能够检测而且能够跟踪欺骗攻击，从而确保可靠且值得信赖的水印检测。在多个数据集和语言模型上进行的广泛实验表明，DualGuard实现了出色的检测性、稳健性、可追溯性和文本质量，有效地提高了现实世界应用程序的LLM水印状态。



## **8. Bounty Hunter: Autonomous, Comprehensive Emulation of Multi-Faceted Adversaries**

赏金猎人：多面对手的自主、全面模拟 cs.CR

15 pages, 9 figures

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15275v1) [paper-pdf](https://arxiv.org/pdf/2512.15275v1)

**Authors**: Louis Hackländer-Jansen, Rafael Uetz, Martin Henze

**Abstract**: Adversary emulation is an essential procedure for cybersecurity assessments such as evaluating an organization's security posture or facilitating structured training and research in dedicated environments. To allow for systematic and time-efficient assessments, several approaches from academia and industry have worked towards the automation of adversarial actions. However, they exhibit significant limitations regarding autonomy, tactics coverage, and real-world applicability. Consequently, adversary emulation remains a predominantly manual task requiring substantial human effort and security expertise - even amidst the rise of Large Language Models. In this paper, we present Bounty Hunter, an automated adversary emulation method, designed and implemented as an open-source plugin for the popular adversary emulation platform Caldera, that enables autonomous emulation of adversaries with multi-faceted behavior while providing a wide coverage of tactics. To this end, it realizes diverse adversarial behavior, such as different levels of detectability and varying attack paths across repeated emulations. By autonomously compromising a simulated enterprise network, Bounty Hunter showcases its ability to achieve given objectives without prior knowledge of its target, including pre-compromise, initial compromise, and post-compromise attack tactics. Overall, Bounty Hunter facilitates autonomous, comprehensive, and multi-faceted adversary emulation to help researchers and practitioners in performing realistic and time-efficient security assessments, training exercises, and intrusion detection research.

摘要: Adobile仿真是网络安全评估的重要程序，例如评估组织的安全态势或促进专用环境中的结构化培训和研究。为了进行系统性且高效的评估，学术界和工业界的多种方法致力于对抗行动的自动化。然而，它们在自主性、战术覆盖范围和现实世界的适用性方面表现出显着的局限性。因此，对手模拟仍然是一项主要的手动任务，需要大量的人力和安全专业知识--即使在大型语言模型的兴起下也是如此。在本文中，我们介绍了Bounty Hunter，这是一种自动化的对手模拟方法，作为流行的对手模拟平台Caldera的开源插件设计和实现，它能够自主模拟具有多方面行为的对手，同时提供广泛的战术覆盖范围。为此，它实现了多样化的对抗行为，例如不同级别的可检测性和重复模拟中的不同攻击路径。通过自主破坏模拟企业网络，Bounty Hunter展示了其在不了解其目标的情况下实现给定目标的能力，包括破坏前、初始破坏和破坏后攻击策略。总体而言，Bounty Hunter促进了自主、全面和多方面的对手模拟，以帮助研究人员和从业者执行现实且省时的安全评估、培训练习和入侵检测研究。



## **9. MCP-SafetyBench: A Benchmark for Safety Evaluation of Large Language Models with Real-World MCP Servers**

MCP-SafetyBench：使用现实世界的LCP服务器对大型语言模型进行安全评估的基准 cs.CL

Our benchmark is available at https://github.com/xjzzzzzzzz/MCPSafety

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15163v1) [paper-pdf](https://arxiv.org/pdf/2512.15163v1)

**Authors**: Xuanjun Zong, Zhiqi Shen, Lei Wang, Yunshi Lan, Chao Yang

**Abstract**: Large language models (LLMs) are evolving into agentic systems that reason, plan, and operate external tools. The Model Context Protocol (MCP) is a key enabler of this transition, offering a standardized interface for connecting LLMs with heterogeneous tools and services. Yet MCP's openness and multi-server workflows introduce new safety risks that existing benchmarks fail to capture, as they focus on isolated attacks or lack real-world coverage. We present MCP-SafetyBench, a comprehensive benchmark built on real MCP servers that supports realistic multi-turn evaluation across five domains: browser automation, financial analysis, location navigation, repository management, and web search. It incorporates a unified taxonomy of 20 MCP attack types spanning server, host, and user sides, and includes tasks requiring multi-step reasoning and cross-server coordination under uncertainty. Using MCP-SafetyBench, we systematically evaluate leading open- and closed-source LLMs, revealing large disparities in safety performance and escalating vulnerabilities as task horizons and server interactions grow. Our results highlight the urgent need for stronger defenses and establish MCP-SafetyBench as a foundation for diagnosing and mitigating safety risks in real-world MCP deployments.

摘要: 大型语言模型（LLM）正在演变为推理、计划和操作外部工具的代理系统。模型上下文协议（MCP）是这种转变的关键推动者，它提供了一个标准化的接口，用于连接LLM与异构工具和服务。然而，MCP的开放性和多服务器工作流程引入了现有基准无法捕获的新安全风险，因为它们专注于孤立的攻击或缺乏真实世界的覆盖。我们提出了MCP-SafetyBench，这是一个建立在真实MCP服务器上的综合基准测试，支持跨五个领域的真实多轮评估：浏览器自动化，财务分析，位置导航，存储库管理和Web搜索。它融合了跨越服务器、主机和用户端的20种HCP攻击类型的统一分类，并包括需要多步推理和不确定性下跨服务器协调的任务。使用MCP-SafetyBench，我们系统地评估领先的开源和开源LLM，揭示了随着任务视野和服务器交互的增长，安全性能的巨大差异和不断升级的漏洞。我们的结果凸显了迫切需要更强大的防御，并将MCP-SafetyBench建立为诊断和缓解现实世界中的LCP部署中安全风险的基础。



## **10. Quantifying Return on Security Controls in LLM Systems**

量化LLM系统中安全控制的回报 cs.CR

13 pages, 9 figures, 3 tables

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15081v1) [paper-pdf](https://arxiv.org/pdf/2512.15081v1)

**Authors**: Richard Helder Moulton, Austin O'Brien, John D. Hastings

**Abstract**: Although large language models (LLMs) are increasingly used in security-critical workflows, practitioners lack quantitative guidance on which safeguards are worth deploying. This paper introduces a decision-oriented framework and reproducible methodology that together quantify residual risk, convert adversarial probe outcomes into financial risk estimates and return-on-control (RoC) metrics, and enable monetary comparison of layered defenses for LLM-based systems. A retrieval-augmented generation (RAG) service is instantiated using the DeepSeek-R1 model over a corpus containing synthetic personally identifiable information (PII), and subjected to automated attacks with Garak across five vulnerability classes: PII leakage, latent context injection, prompt injection, adversarial attack generation, and divergence. For each (vulnerability, control) pair, attack success probabilities are estimated via Laplace's Rule of Succession and combined with loss triangle distributions, calibrated from public breach-cost data, in 10,000-run Monte Carlo simulations to produce loss exceedance curves and expected losses. Three widely used mitigations, attribute-based access control (ABAC); named entity recognition (NER) redaction using Microsoft Presidio; and NeMo Guardrails, are then compared to a baseline RAG configuration. The baseline system exhibits very high attack success rates (>= 0.98 for PII, latent injection, and prompt injection), yielding a total simulated expected loss of $313k per attack scenario. ABAC collapses success probabilities for PII and prompt-related attacks to near zero and reduces the total expected loss by ~94%, achieving an RoC of 9.83. NER redaction likewise eliminates PII leakage and attains an RoC of 5.97, while NeMo Guardrails provides only marginal benefit (RoC of 0.05).

摘要: 尽管大型语言模型（LLM）越来越多地用于安全关键工作流程，但从业者缺乏关于哪些保障措施值得部署的量化指导。本文介绍了一个面向决策的框架和可重复的方法论，它们共同量化剩余风险，将对抗性调查结果转化为财务风险估计和控制回报（RoC）指标，并实现基于LLM的系统的分层防御的货币比较。检索增强生成（RAG）服务使用DeepSeek-R1模型在包含合成个人可识别信息（PRI）的数据库上实例化，并在五个漏洞类别上受到Garak的自动攻击：PIP泄露、潜在上下文注入、提示注入、对抗攻击生成和分歧。对于每个（漏洞、控制）对，攻击成功概率是通过拉普拉斯继承规则估计的，并结合根据公共违规成本数据校准的损失三角分布，在10，000次运行的蒙特卡洛模拟中生成损失延迟曲线和预期损失。然后将三种广泛使用的缓解措施：基于属性的访问控制（ABAC）;使用Microsoft Presidio的命名实体识别（NER）编辑;和NeMo Guardrails与基线RAG配置进行比较。基线系统表现出非常高的攻击成功率（PRI、潜伏注射和即时注射>= 0.98），每个攻击场景的模拟预期损失总额为31.3万美元。ABAC将PRI和预算相关攻击的成功概率降至接近零，并将总预期损失降低约94%，实现了9.83的RoC。NER编辑同样消除了PRI泄漏，并获得了5.97的RoC，而NeMo Guardrails仅提供了边际效益（RoC为0.05）。



## **11. MALCDF: A Distributed Multi-Agent LLM Framework for Real-Time Cyber**

MALEDF：用于实时网络的分布式多代理LLM框架 cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14846v1) [paper-pdf](https://arxiv.org/pdf/2512.14846v1)

**Authors**: Arth Bhardwaj, Sia Godika, Yuvam Loonker

**Abstract**: Traditional, centralized security tools often miss adaptive, multi-vector attacks. We present the Multi-Agent LLM Cyber Defense Framework (MALCDF), a practical setup where four large language model (LLM) agents-Detection, Intelligence, Response, and Analysis-work together in real time. Agents communicate over a Secure Communication Layer (SCL) with encrypted, ontology-aligned messages, and produce audit-friendly outputs (e.g., MITRE ATT&CK mappings).   For evaluation, we keep the test simple and consistent: all reported metrics come from the same 50-record live stream derived from the CICIDS2017 feature schema. CICIDS2017 is used for configuration (fields/schema) and to train a practical ML baseline. The ML-IDS baseline is a Lightweight Random Forest IDS (LRF-IDS) trained on a subset of CICIDS2017 and tested on the 50-record stream, with no overlap between training and test records.   In experiments, MALCDF reaches 90.0% detection accuracy, 85.7% F1-score, and 9.1% false-positive rate, with 6.8s average per-event latency. It outperforms the lightweight ML-IDS baseline and a single-LLM setup on accuracy while keeping end-to-end outputs consistent. Overall, this hands-on build suggests that coordinating simple LLM agents with secure, ontology-aligned messaging can improve practical, real-time cyber defense.

摘要: 传统的集中式安全工具经常错过自适应的多载体攻击。我们介绍了多代理LLM网络防御框架（MALEDF），这是一种实用的设置，四个大型语言模型（LLM）代理--检测、情报、响应和分析--实时协同工作。代理通过安全通信层（SCL）与加密的、与实体对齐的消息进行通信，并产生对用户友好的输出（例如，MITRE ATT & CK映射）。   对于评估，我们保持测试简单且一致：所有报告的指标都来自源自CICIDS 2017功能模式的相同50条记录直播流。CICIDS 2017用于配置（字段/模式）和训练实用的ML基线。ML-IDS基线是轻量级随机森林IDS（LRF-IDS），在CICIDS 2017的子集上训练，并在50条记录流上进行测试，训练和测试记录之间没有重叠。   实验中，MALEDF的检测准确率达到90.0%，F1评分达到85.7%，假阳性率达到9.1%，平均每事件潜伏期为6.8s。它在准确性方面优于轻量级ML-IDS基线和单LLM设置，同时保持端到端输出一致。总体而言，这种动手构建表明，将简单的LLM代理与安全的、与实体一致的消息传递协调起来可以改善实用的实时网络防御。



## **12. PerProb: Indirectly Evaluating Memorization in Large Language Models**

PerProb：间接评估大型语言模型中的精简化 cs.CR

Accepted at APSEC 2025

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14600v1) [paper-pdf](https://arxiv.org/pdf/2512.14600v1)

**Authors**: Yihan Liao, Jacky Keung, Xiaoxue Ma, Jingyu Zhang, Yicheng Sun

**Abstract**: The rapid advancement of Large Language Models (LLMs) has been driven by extensive datasets that may contain sensitive information, raising serious privacy concerns. One notable threat is the Membership Inference Attack (MIA), where adversaries infer whether a specific sample was used in model training. However, the true impact of MIA on LLMs remains unclear due to inconsistent findings and the lack of standardized evaluation methods, further complicated by the undisclosed nature of many LLM training sets. To address these limitations, we propose PerProb, a unified, label-free framework for indirectly assessing LLM memorization vulnerabilities. PerProb evaluates changes in perplexity and average log probability between data generated by victim and adversary models, enabling an indirect estimation of training-induced memory. Compared with prior MIA methods that rely on member/non-member labels or internal access, PerProb is independent of model and task, and applicable in both black-box and white-box settings. Through a systematic classification of MIA into four attack patterns, we evaluate PerProb's effectiveness across five datasets, revealing varying memory behaviors and privacy risks among LLMs. Additionally, we assess mitigation strategies, including knowledge distillation, early stopping, and differential privacy, demonstrating their effectiveness in reducing data leakage. Our findings offer a practical and generalizable framework for evaluating and improving LLM privacy.

摘要: 大型语言模型（LLM）的快速发展是由可能包含敏感信息的大量数据集推动的，从而引发了严重的隐私问题。一个值得注意的威胁是会员推断攻击（MIA），对手推断特定样本是否用于模型训练。然而，由于调查结果不一致和缺乏标准化的评估方法，MIA对LLM的真正影响仍然不清楚，而且许多LLM培训集的未公开性质使其更加复杂。为了解决这些限制，我们提出了PerProb，这是一个统一的、无标签的框架，用于间接评估LLM记忆漏洞。PerProb评估受害者和对手模型生成的数据之间的困惑度和平均日志概率的变化，从而能够间接估计训练诱导的记忆。与之前依赖成员/非成员标签或内部访问的MIA方法相比，PerProb独立于模型和任务，适用于黑盒和白盒设置。通过将MIA系统地分类为四种攻击模式，我们评估了PerProb在五个数据集中的有效性，揭示了LLM之间不同的记忆行为和隐私风险。此外，我们还评估了缓解策略，包括知识提炼、提前停止和差异隐私，证明它们在减少数据泄露方面的有效性。我们的研究结果为评估和改善LLM隐私提供了一个实用和可推广的框架。



## **13. Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space**

通过隐形方式转移对LLM代理的推理方式中毒：RSV空间中的过程级攻击和收件箱监控 cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14448v1) [paper-pdf](https://arxiv.org/pdf/2512.14448v1)

**Authors**: Xingfu Zhou, Pengfei Wang

**Abstract**: Large Language Model (LLM) agents relying on external retrieval are increasingly deployed in high-stakes environments. While existing adversarial attacks primarily focus on content falsification or instruction injection, we identify a novel, process-oriented attack surface: the agent's reasoning style. We propose Reasoning-Style Poisoning (RSP), a paradigm that manipulates how agents process information rather than what they process. We introduce Generative Style Injection (GSI), an attack method that rewrites retrieved documents into pathological tones--specifically "analysis paralysis" or "cognitive haste"--without altering underlying facts or using explicit triggers. To quantify these shifts, we develop the Reasoning Style Vector (RSV), a metric tracking Verification depth, Self-confidence, and Attention focus. Experiments on HotpotQA and FEVER using ReAct, Reflection, and Tree of Thoughts (ToT) architectures reveal that GSI significantly degrades performance. It increases reasoning steps by up to 4.4 times or induces premature errors, successfully bypassing state-of-the-art content filters. Finally, we propose RSP-M, a lightweight runtime monitor that calculates RSV metrics in real-time and triggers alerts when values exceed safety thresholds. Our work demonstrates that reasoning style is a distinct, exploitable vulnerability, necessitating process-level defenses beyond static content analysis.

摘要: 依赖外部检索的大型语言模型（LLM）代理越来越多地部署在高风险环境中。虽然现有的对抗性攻击主要集中在内容伪造或指令注入上，但我们发现了一种新颖的、面向过程的攻击表面：代理的推理风格。我们提出了推理式中毒（RSP），这是一种操纵代理如何处理信息而不是处理内容的范式。我们引入了生成风格注入（GSI），这是一种攻击方法，将检索到的文档改写为病理性语气--特别是“分析瘫痪”或“认知仓促”--而无需改变基本事实或使用明确的触发器。为了量化这些转变，我们开发了推理风格载体（RSV），这是一种跟踪验证深度、自信和注意力焦点的指标。使用ReAct、ReReReflection和Tree of Thoughts（ToT）架构对HotpotQA和FEVER进行的实验表明，GSI会显着降低性能。它将推理步骤增加多达4.4倍，否则会导致过早错误，从而成功绕过最先进的内容过滤器。最后，我们提出了RSP-M，这是一种轻量级的运行时监视器，可实时计算RSV指标，并在值超过安全阈值时触发警报。我们的工作表明，推理风格是一个独特的、可利用的漏洞，需要静态内容分析之外的流程级防御。



## **14. Semantic Mismatch and Perceptual Degradation: A New Perspective on Image Editing Immunity**

语义不匹配和感知退化：图像编辑免疫的新视角 cs.CV

11 pages, 4 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14320v1) [paper-pdf](https://arxiv.org/pdf/2512.14320v1)

**Authors**: Shuai Dong, Jie Zhang, Guoying Zhao, Shiguang Shan, Xilin Chen

**Abstract**: Text-guided image editing via diffusion models, while powerful, raises significant concerns about misuse, motivating efforts to immunize images against unauthorized edits using imperceptible perturbations. Prevailing metrics for evaluating immunization success typically rely on measuring the visual dissimilarity between the output generated from a protected image and a reference output generated from the unprotected original. This approach fundamentally overlooks the core requirement of image immunization, which is to disrupt semantic alignment with attacker intent, regardless of deviation from any specific output. We argue that immunization success should instead be defined by the edited output either semantically mismatching the prompt or suffering substantial perceptual degradations, both of which thwart malicious intent. To operationalize this principle, we propose Synergistic Intermediate Feature Manipulation (SIFM), a method that strategically perturbs intermediate diffusion features through dual synergistic objectives: (1) maximizing feature divergence from the original edit trajectory to disrupt semantic alignment with the expected edit, and (2) minimizing feature norms to induce perceptual degradations. Furthermore, we introduce the Immunization Success Rate (ISR), a novel metric designed to rigorously quantify true immunization efficacy for the first time. ISR quantifies the proportion of edits where immunization induces either semantic failure relative to the prompt or significant perceptual degradations, assessed via Multimodal Large Language Models (MLLMs). Extensive experiments show our SIFM achieves the state-of-the-art performance for safeguarding visual content against malicious diffusion-based manipulation.

摘要: 通过扩散模型进行文本引导的图像编辑虽然功能强大，但引起了对滥用的严重关注，促使人们努力使用不可察觉的扰动使图像免受未经授权的编辑。用于评估免疫成功的流行度量通常依赖于测量从受保护图像生成的输出与从未受保护原始图像生成的参考输出之间的视觉不相似性。这种方法从根本上忽视了图像免疫的核心要求，即破坏与攻击者意图的语义一致，无论是否偏离任何特定输出。我们认为，免疫成功应该由编辑后的输出来定义，要么在语义上与提示不匹配，要么遭受严重的感知退化，这两者都会阻止恶意意图。为了实现这一原则，我们提出了协同中间特征操作（SIFM），这是一种通过双重协同目标战略性地干扰中间扩散特征的方法：（1）最大化与原始编辑轨迹的特征分歧，以破坏与预期编辑的语义对齐，以及（2）最小化特征规范，以诱导感知退化。此外，我们还引入了免疫成功率（ISR），这是一种旨在首次严格量化真实免疫效力的新指标。ISR量化了免疫诱导语义失败相对于提示或显著感知退化的编辑比例，通过多模态大语言模型（MLLM）进行评估。大量的实验表明，我们的SIFM实现了最先进的性能，以保护视觉内容免受恶意的基于扩散的操纵。



## **15. PentestEval: Benchmarking LLM-based Penetration Testing with Modular and Stage-Level Design**

PentestEval：通过模块化和阶段级设计对基于LLM的渗透测试进行基准测试 cs.SE

13 pages, 6 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14233v1) [paper-pdf](https://arxiv.org/pdf/2512.14233v1)

**Authors**: Ruozhao Yang, Mingfei Cheng, Gelei Deng, Tianwei Zhang, Junjie Wang, Xiaofei Xie

**Abstract**: Penetration testing is essential for assessing and strengthening system security against real-world threats, yet traditional workflows remain highly manual, expertise-intensive, and difficult to scale. Although recent advances in Large Language Models (LLMs) offer promising opportunities for automation, existing applications rely on simplistic prompting without task decomposition or domain adaptation, resulting in unreliable black-box behavior and limited insight into model capabilities across penetration testing stages. To address this gap, we introduce PentestEval, the first comprehensive benchmark for evaluating LLMs across six decomposed penetration testing stages: Information Collection, Weakness Gathering and Filtering, Attack Decision-Making, Exploit Generation and Revision. PentestEval integrates expert-annotated ground truth with a fully automated evaluation pipeline across 346 tasks covering all stages in 12 realistic vulnerable scenarios. Our stage-level evaluation of 9 widely used LLMs reveals generally weak performance and distinct limitations across the stages of penetration-testing workflow. End-to-end pipelines reach only 31% success rate, and existing LLM-powered systems such as PentestGPT, PentestAgent, and VulnBot exhibit similar limitations, with autonomous agents failing almost entirely. These findings highlight that autonomous penetration testing demands stronger structured reasoning, where modularization enhances each individual stage and improves overall performance. PentestEval provides the foundational benchmark needed for future research on fine-grained, stage-level evaluation, paving the way toward more reliable LLM-based automation.

摘要: 渗透测试对于评估和加强系统安全性以应对现实世界的威胁至关重要，但传统工作流程仍然高度手动、专业知识密集且难以扩展。尽管大型语言模型（LLM）的最新进展为自动化提供了有希望的机会，但现有的应用程序依赖于简单化的提示，而没有任务分解或域适应，导致黑匣子行为不可靠，并且对渗透测试阶段模型能力的洞察有限。为了弥补这一差距，我们引入了PentestEval，这是第一个用于评估跨六个分解渗透测试阶段LLM的综合基准：信息收集、弱点收集和过滤、攻击决策、漏洞利用生成和修订。PentestEval将专家注释的基本真相与全自动评估管道集成，涵盖346项任务，涵盖12个现实脆弱场景的所有阶段。我们对9个广泛使用的LLM的阶段级评估显示，渗透测试工作流程各个阶段的性能普遍较弱，并且存在明显的局限性。端到端管道的成功率仅为31%，而现有的LLM驱动系统（例如PentestGPT、PentestAgent和VulnBot）也表现出类似的局限性，自主代理几乎完全失败。这些发现凸显了自主渗透测试需要更强的结构化推理，其中模块化增强了每个单独阶段并提高了整体性能。PentestEval为未来细粒度、阶段级评估研究提供了所需的基础基准，为更可靠的基于LLM的自动化铺平了道路。



## **16. IntentMiner: Intent Inversion Attack via Tool Call Analysis in the Model Context Protocol**

IntentMiner：通过模型上下文协议中的工具调用分析进行意图倒置攻击 cs.CR

12 pages, 6 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14166v1) [paper-pdf](https://arxiv.org/pdf/2512.14166v1)

**Authors**: Yunhao Yao, Zhiqiang Wang, Haoran Cheng, Yihang Cheng, Haohua Du, Xiang-Yang Li

**Abstract**: The rapid evolution of Large Language Models (LLMs) into autonomous agents has led to the adoption of the Model Context Protocol (MCP) as a standard for discovering and invoking external tools. While this architecture decouples the reasoning engine from tool execution to enhance scalability, it introduces a significant privacy surface: third-party MCP servers, acting as semi-honest intermediaries, can observe detailed tool interaction logs outside the user's trusted boundary. In this paper, we first identify and formalize a novel privacy threat termed Intent Inversion, where a semi-honest MCP server attempts to reconstruct the user's private underlying intent solely by analyzing legitimate tool calls. To systematically assess this vulnerability, we propose IntentMiner, a framework that leverages Hierarchical Information Isolation and Three-Dimensional Semantic Analysis, integrating tool purpose, call statements, and returned results, to accurately infer user intent at the step level. Extensive experiments demonstrate that IntentMiner achieves a high degree of semantic alignment (over 85%) with original user queries, significantly outperforming baseline approaches. These results highlight the inherent privacy risks in decoupled agent architectures, revealing that seemingly benign tool execution logs can serve as a potent vector for exposing user secrets.

摘要: 大型语言模型（LLM）快速演变为自治代理，导致模型上下文协议（HCP）被采用作为发现和调用外部工具的标准。虽然该架构将推理引擎与工具执行分开以增强可扩展性，但它引入了一个重要的隐私表面：充当半诚实中介的第三方LCP服务器可以观察用户可信边界之外的详细工具交互日志。在本文中，我们首先识别并正式化一种名为“意图倒置”的新型隐私威胁，其中半诚实的LCP服务器尝试仅通过分析合法的工具调用来重建用户的私人底层意图。为了系统性地评估此漏洞，我们提出了IntentMiner，这是一个利用分层信息隔离和三维语义分析的框架，集成工具目的、调用陈述和返回的结果，可以在步骤级别准确地推断用户意图。大量实验表明，IntentMiner与原始用户查询实现了高度的语义一致性（超过85%），显着优于基线方法。这些结果凸显了脱钩代理架构中固有的隐私风险，揭示了看似良性的工具执行日志可以作为暴露用户秘密的有力载体。



## **17. From Obfuscated to Obvious: A Comprehensive JavaScript Deobfuscation Tool for Security Analysis**

从混淆到明显：用于安全分析的全面JavaScript去混淆工具 cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14070v1) [paper-pdf](https://arxiv.org/pdf/2512.14070v1)

**Authors**: Dongchao Zhou, Lingyun Ying, Huajun Chai, Dongbin Wang

**Abstract**: JavaScript's widespread adoption has made it an attractive target for malicious attackers who employ sophisticated obfuscation techniques to conceal harmful code. Current deobfuscation tools suffer from critical limitations that severely restrict their practical effectiveness. Existing tools struggle with diverse input formats, address only specific obfuscation types, and produce cryptic output that impedes human analysis.   To address these challenges, we present JSIMPLIFIER, a comprehensive deobfuscation tool using a multi-stage pipeline with preprocessing, abstract syntax tree-based static analysis, dynamic execution tracing, and Large Language Model (LLM)-enhanced identifier renaming. We also introduce multi-dimensional evaluation metrics that integrate control/data flow analysis, code simplification assessment, entropy measures and LLM-based readability assessments.   We construct and release the largest real-world obfuscated JavaScript dataset with 44,421 samples (23,212 wild malicious + 21,209 benign samples). Evaluation shows JSIMPLIFIER outperforms existing tools with 100% processing capability across 20 obfuscation techniques, 100% correctness on evaluation subsets, 88.2% code complexity reduction, and over 4-fold readability improvement validated by multiple LLMs. Our results advance benchmarks for JavaScript deobfuscation research and practical security applications.

摘要: JavaScript的广泛采用使其成为恶意攻击者的一个有吸引力的目标，这些攻击者使用复杂的混淆技术来隐藏有害代码。当前的去模糊工具存在严重局限性，严重限制了其实际有效性。现有的工具难以应对不同的输入格式，仅解决特定的混淆类型，并产生阻碍人类分析的神秘输出。   为了应对这些挑战，我们提出了JSIMPPLIFIER，这是一个全面的去模糊工具，使用多阶段管道，具有预处理、基于抽象语法树的静态分析、动态执行跟踪和大型语言模型（LLM）增强的标识符重命名。我们还介绍了多维的评估指标，集成控制/数据流分析，代码简化评估，熵的措施和基于LLM的可读性评估。   我们构建并发布了最大的真实世界混淆JavaScript数据集，包含44，421个样本（23，212个野生恶意样本+21，209个良性样本）。评估显示，JSIMPLIFIER优于现有工具，具有20种模糊技术100%的处理能力、评估子集的正确性100%、代码复杂性降低88.2%以及经多个LLM验证的4倍以上的可读性改进。我们的结果推进了JavaScript去模糊研究和实际安全应用的基准测试。



## **18. Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures (XAMT)**

异类多代理体系结构（XAMT）中隐蔽内存篡改的双层优化 cs.CR

10 pages, 5 figures, 4 tables. Conference-style paper (IEEEtran). Proposes unified bilevel optimization framework for covert memory poisoning attacks in heterogeneous multi-agent systems (MARL + RAG)

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.15790v1) [paper-pdf](https://arxiv.org/pdf/2512.15790v1)

**Authors**: Akhil Sharma, Shaikh Yaser Arafat, Jai Kumar Sharma, Ken Huang

**Abstract**: The increasing operational reliance on complex Multi-Agent Systems (MAS) across safety-critical domains necessitates rigorous adversarial robustness assessment. Modern MAS are inherently heterogeneous, integrating conventional Multi-Agent Reinforcement Learning (MARL) with emerging Large Language Model (LLM) agent architectures utilizing Retrieval-Augmented Generation (RAG). A critical shared vulnerability is reliance on centralized memory components: the shared Experience Replay (ER) buffer in MARL and the external Knowledge Base (K) in RAG agents. This paper proposes XAMT (Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures), a novel framework that formalizes attack generation as a bilevel optimization problem. The Upper Level minimizes perturbation magnitude (delta) to enforce covertness while maximizing system behavior divergence toward an adversary-defined target (Lower Level). We provide rigorous mathematical instantiations for CTDE MARL algorithms and RAG-based LLM agents, demonstrating that bilevel optimization uniquely crafts stealthy, minimal-perturbation poisons evading detection heuristics. Comprehensive experimental protocols utilize SMAC and SafeRAG benchmarks to quantify effectiveness at sub-percent poison rates (less than or equal to 1 percent in MARL, less than or equal to 0.1 percent in RAG). XAMT defines a new unified class of training-time threats essential for developing intrinsically secure MAS, with implications for trust, formal verification, and defensive strategies prioritizing intrinsic safety over perimeter-based detection.

摘要: 安全关键领域对复杂多智能体系统（MAS）的运营依赖日益增加，需要进行严格的对抗稳健性评估。现代MAS本质上是异类的，将传统的多智能体强化学习（MARL）与利用检索增强生成（RAG）的新兴大型语言模型（LLM）智能体架构集成在一起。一个关键的共享漏洞是对集中式内存组件的依赖：MARL中的共享体验重播（ER）缓冲区和RAG代理中的外部知识库（K）。本文提出了XAPT（在异类多代理体系结构中针对隐蔽内存篡改的双层优化），这是一个新颖的框架，将攻击生成形式化为双层优化问题。上级最小化扰动幅度（增量）以加强隐蔽性，同时最大化系统行为向敌对定义的目标（下级）的分歧。我们为CTDE MARL算法和基于RAG的LLM代理提供了严格的数学实例，证明两层优化独特地处理了逃避检测启发的隐形、最小扰动毒药。全面的实验方案利用SMAC和SafeRAG基准来量化次百分中毒率（MARL中小于或等于1%，RAG中小于或等于0.1%）的有效性。XAMT定义了一种新的统一训练时威胁，这对于开发本质安全的MAS至关重要，对信任、形式验证和防御策略产生了影响，这些策略优先于本质安全而不是基于边界的检测。



## **19. On the Effectiveness of Membership Inference in Targeted Data Extraction from Large Language Models**

隶属推理在大型语言模型有针对性数据提取中的有效性 cs.LG

Accepted to IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) 2026

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13352v1) [paper-pdf](https://arxiv.org/pdf/2512.13352v1)

**Authors**: Ali Al Sahili, Ali Chehab, Razane Tajeddine

**Abstract**: Large Language Models (LLMs) are prone to memorizing training data, which poses serious privacy risks. Two of the most prominent concerns are training data extraction and Membership Inference Attacks (MIAs). Prior research has shown that these threats are interconnected: adversaries can extract training data from an LLM by querying the model to generate a large volume of text and subsequently applying MIAs to verify whether a particular data point was included in the training set. In this study, we integrate multiple MIA techniques into the data extraction pipeline to systematically benchmark their effectiveness. We then compare their performance in this integrated setting against results from conventional MIA benchmarks, allowing us to evaluate their practical utility in real-world extraction scenarios.

摘要: 大型语言模型（LLM）容易记住训练数据，这会带来严重的隐私风险。两个最突出的问题是训练数据提取和成员推断攻击（MIA）。之前的研究表明，这些威胁是相互关联的：对手可以通过查询模型以生成大量文本，然后应用MIA来验证特定数据点是否包含在训练集中，从而从LLM中提取训练数据。在这项研究中，我们将多种MIA技术集成到数据提取管道中，以系统地衡量其有效性。然后，我们将它们在此集成环境中的性能与传统MIA基准的结果进行比较，使我们能够评估它们在现实世界提取场景中的实际实用性。



## **20. Cisco Integrated AI Security and Safety Framework Report**

思科集成人工智能安全和安全框架报告 cs.CR

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12921v1) [paper-pdf](https://arxiv.org/pdf/2512.12921v1)

**Authors**: Amy Chang, Tiffany Saade, Sanket Mendapara, Adam Swanda, Ankit Garg

**Abstract**: Artificial intelligence (AI) systems are being readily and rapidly adopted, increasingly permeating critical domains: from consumer platforms and enterprise software to networked systems with embedded agents. While this has unlocked potential for human productivity gains, the attack surface has expanded accordingly: threats now span content safety failures (e.g., harmful or deceptive outputs), model and data integrity compromise (e.g., poisoning, supply-chain tampering), runtime manipulations (e.g., prompt injection, tool and agent misuse), and ecosystem risks (e.g., orchestration abuse, multi-agent collusion). Existing frameworks such as MITRE ATLAS, National Institute of Standards and Technology (NIST) AI 100-2 Adversarial Machine Learning (AML) taxonomy, and OWASP Top 10s for Large Language Models (LLMs) and Agentic AI Applications provide valuable viewpoints, but each covers only slices of this multi-dimensional space.   This paper presents Cisco's Integrated AI Security and Safety Framework ("AI Security Framework"), a unified, lifecycle-aware taxonomy and operationalization framework that can be used to classify, integrate, and operationalize the full range of AI risks. It integrates AI security and AI safety across modalities, agents, pipelines, and the broader ecosystem. The AI Security Framework is designed to be practical for threat identification, red-teaming, risk prioritization, and it is comprehensive in scope and can be extensible to emerging deployments in multimodal contexts, humanoids, wearables, and sensory infrastructures. We analyze gaps in prevailing frameworks, discuss design principles for our framework, and demonstrate how the taxonomy provides structure for understanding how modern AI systems fail, how adversaries exploit these failures, and how organizations can build defenses across the AI lifecycle that evolve alongside capability advancements.

摘要: 人工智能（AI）系统正在被轻松而快速地采用，并日益渗透到关键领域：从消费者平台和企业软件到具有嵌入式代理的网络系统。虽然这释放了人类生产力提高的潜力，但攻击面也相应扩大：威胁现在跨越内容安全故障（例如，有害或欺骗性输出）、模型和数据完整性损害（例如，中毒、供应链篡改）、运行时操纵（例如，及时注射、工具和试剂滥用）和生态系统风险（例如，编排滥用、多代理勾结）。MITRE ATLAS、美国国家标准与技术研究院（NIH）AI 100-2对抗性机器学习（ML）分类法以及OWASP大型语言模型（LLM）和统计性人工智能应用程序十大框架提供了有价值的观点，但每个框架都只涵盖了这个多维空间的一部分。   本文介绍了思科的集成人工智能安全框架（“人工智能安全框架”），这是一个统一的、生命周期感知的分类和操作框架，可用于分类、集成和操作全方位人工智能风险。它集成了人工智能安全和跨模式、代理、管道和更广泛生态系统的人工智能安全。人工智能安全框架旨在实用于威胁识别、红色分组、风险优先级，而且它的范围全面，可以扩展到多模式环境中的新兴部署、人形机器人、可穿戴设备和感官基础设施。我们分析了主流框架中的差距，讨论了框架的设计原则，并演示了分类法如何提供结构来理解现代人工智能系统如何失败、对手如何利用这些失败，以及组织如何在整个人工智能生命周期中构建防御，并随着能力的进步而发展。



## **21. CTIGuardian: A Few-Shot Framework for Mitigating Privacy Leakage in Fine-Tuned LLMs**

CTIGuardian：一个用于缓解微调LLM中隐私泄露的少镜头框架 cs.CR

Accepted at the 18th Cybersecurity Experimentation and Test Workshop (CSET), in conjunction with ACSAC 2025

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12914v1) [paper-pdf](https://arxiv.org/pdf/2512.12914v1)

**Authors**: Shashie Dilhara Batan Arachchige, Benjamin Zi Hao Zhao, Hassan Jameel Asghar, Dinusha Vatsalan, Dali Kaafar

**Abstract**: Large Language Models (LLMs) are often fine-tuned to adapt their general-purpose knowledge to specific tasks and domains such as cyber threat intelligence (CTI). Fine-tuning is mostly done through proprietary datasets that may contain sensitive information. Owners expect their fine-tuned model to not inadvertently leak this information to potentially adversarial end users. Using CTI as a use case, we demonstrate that data-extraction attacks can recover sensitive information from fine-tuned models on CTI reports, underscoring the need for mitigation. Retraining the full model to eliminate this leakage is computationally expensive and impractical. We propose an alternative approach, which we call privacy alignment, inspired by safety alignment in LLMs. Just like safety alignment teaches the model to abide by safety constraints through a few examples, we enforce privacy alignment through few-shot supervision, integrating a privacy classifier and a privacy redactor, both handled by the same underlying LLM. We evaluate our system, called CTIGuardian, using GPT-4o mini and Mistral-7B Instruct models, benchmarking against Presidio, a named entity recognition (NER) baseline. Results show that CTIGuardian provides a better privacy-utility trade-off than NER based models. While we demonstrate its effectiveness on a CTI use case, the framework is generic enough to be applicable to other sensitive domains.

摘要: 大型语言模型（LLM）通常经过微调，以使其通用知识适应特定任务和领域，例如网络威胁情报（RTI）。微调主要通过可能包含敏感信息的专有数据集完成。所有者希望他们的微调模型不会无意中将此信息泄露给潜在敌对的最终用户。使用RTI作为用例，我们证明数据提取攻击可以从RTI报告上的微调模型中恢复敏感信息，强调了缓解的必要性。重新训练完整模型以消除这种泄漏在计算上昂贵且不切实际。我们提出了一种替代方法，我们称之为隐私对齐，其灵感来自LLM中的安全对齐。就像安全对齐通过几个例子教导模型遵守安全约束一样，我们通过少量监督来强制隐私对齐，集成了隐私分类器和隐私编辑器，两者都由相同的底层LLM处理。我们使用GPT-4 o mini和Mistral-7 B Direct模型评估我们的系统（称为CTIGGuardian），并以Presidio（命名实体识别（NER）基线）为基准。结果表明，CTIGuardian比基于NER的模型提供了更好的隐私与公用事业权衡。虽然我们在RTI用例中证明了其有效性，但该框架足够通用，可以适用于其他敏感领域。



## **22. Auto-Tuning Safety Guardrails for Black-Box Large Language Models**

黑匣子大型语言模型的自动调整安全护栏 cs.CR

8 pages, 7 figures, 1 table. Work completed as part of the M.S. in Artificial Intelligence at the University of St. Thomas using publicly available models and datasets; all views and any errors are the author's own

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.15782v1) [paper-pdf](https://arxiv.org/pdf/2512.15782v1)

**Authors**: Perry Abdulkadir

**Abstract**: Large language models (LLMs) are increasingly deployed behind safety guardrails such as system prompts and content filters, especially in settings where product teams cannot modify model weights. In practice these guardrails are typically hand-tuned, brittle, and difficult to reproduce. This paper studies a simple but practical alternative: treat safety guardrail design itself as a hyperparameter optimization problem over a frozen base model. Concretely, I wrap Mistral-7B-Instruct with modular jailbreak and malware system prompts plus a ModernBERT-based harmfulness classifier, then evaluate candidate configurations on three public benchmarks covering malware generation, classic jailbreak prompts, and benign user queries. Each configuration is scored using malware and jailbreak attack success rate, benign harmful-response rate, and end-to-end latency. A 48-point grid search over prompt combinations and filter modes establishes a baseline. I then run a black-box Optuna study over the same space and show that it reliably rediscovers the best grid configurations while requiring an order of magnitude fewer evaluations and roughly 8x less wall-clock time. The results suggest that viewing safety guardrails as tunable hyperparameters is a feasible way to harden black-box LLM deployments under compute and time constraints.

摘要: 大型语言模型（LLM）越来越多地部署在系统提示和内容过滤器等安全护栏后面，特别是在产品团队无法修改模型权重的环境中。实际上，这些护栏通常是手工调整的、易碎的、难以复制的。本文研究了一种简单但实用的替代方案：将安全护栏设计本身视为冻结基础模型上的超参数优化问题。具体来说，我将Mistral-7 B-Direct与模块化越狱和恶意软件系统提示以及基于ModernBERT的危害性分类器一起包装，然后在涵盖恶意软件生成、经典越狱提示和良性用户查询的三个公共基准上评估候选配置。每个配置都使用恶意软件和越狱攻击成功率、良性有害响应率和端到端延迟进行评分。对提示组合和过滤器模式进行48点网格搜索来建立基线。然后，我在同一空间上运行了一项黑匣子Optuna研究，并表明它可以可靠地重新发现最佳网格配置，同时需要减少一个数量级的评估和大约8倍的壁挂时间。结果表明，将安全护栏视为可调超参数是在计算和时间限制下强化黑匣子LLM部署的可行方法。



## **23. CODE ACROSTIC: Robust Watermarking for Code Generation**

代码ACROSTIC：用于代码生成的鲁棒水印 cs.CR

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.14753v1) [paper-pdf](https://arxiv.org/pdf/2512.14753v1)

**Authors**: Li Lin, Siyuan Xin, Yang Cao, Xiaochun Cao

**Abstract**: Watermarking large language models (LLMs) is vital for preventing their misuse, including the fabrication of fake news, plagiarism, and spam. It is especially important to watermark LLM-generated code, as it often contains intellectual property.However, we found that existing methods for watermarking LLM-generated code fail to address comment removal attack.In such cases, an attacker can simply remove the comments from the generated code without affecting its functionality, significantly reducing the effectiveness of current code-watermarking techniques.On the other hand, injecting a watermark into code is challenging because, as previous works have noted, most code represents a low-entropy scenario compared to natural language. Our approach to addressing this issue involves leveraging prior knowledge to distinguish between low-entropy and high-entropy parts of the code, as indicated by a Cue List of words.We then inject the watermark guided by this Cue List, achieving higher detectability and usability than existing methods.We evaluated our proposed method on HumanEvaland compared our method with three state-of-the-art code watermarking techniques. The results demonstrate the effectiveness of our approach.

摘要: 对大型语言模型（LLM）进行水印对于防止其滥用（包括捏造假新闻、抄袭和垃圾邮件）至关重要。对LLM生成的代码进行水印尤其重要，因为它通常包含知识产权。然而，我们发现对LLM生成的代码进行水印的现有方法无法解决评论删除攻击。在这种情况下，攻击者可以简单地从生成的代码中删除评论而不会影响其功能，从而显着降低了当前代码水印技术的有效性。另一方面，将水印注入到代码中是具有挑战性的，因为正如之前的作品所指出的那样，与自然语言相比，大多数代码代表了一种低熵的场景。我们解决这个问题的方法涉及利用先验知识来区分代码的低信息和高信息部分，如单词的提示列表所示。然后我们注入由该提示列表引导的水印，实现比现有方法更高的可检测性和可用性。我们在HumanEvaland上评估了我们提出的方法，将我们的方法与三种最先进的代码水印技术进行了比较。结果证明了我们方法的有效性。



## **24. The Laminar Flow Hypothesis: Detecting Jailbreaks via Semantic Turbulence in Large Language Models**

流假说：通过大型语言模型中的语义湍流检测越狱 cs.LG

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.13741v1) [paper-pdf](https://arxiv.org/pdf/2512.13741v1)

**Authors**: Md. Hasib Ur Rahman

**Abstract**: As Large Language Models (LLMs) become ubiquitous, the challenge of securing them against adversarial "jailbreaking" attacks has intensified. Current defense strategies often rely on computationally expensive external classifiers or brittle lexical filters, overlooking the intrinsic dynamics of the model's reasoning process. In this work, the Laminar Flow Hypothesis is introduced, which posits that benign inputs induce smooth, gradual transitions in an LLM's high-dimensional latent space, whereas adversarial prompts trigger chaotic, high-variance trajectories - termed Semantic Turbulence - resulting from the internal conflict between safety alignment and instruction-following objectives. This phenomenon is formalized through a novel, zero-shot metric: the variance of layer-wise cosine velocity. Experimental evaluation across diverse small language models reveals a striking diagnostic capability. The RLHF-aligned Qwen2-1.5B exhibits a statistically significant 75.4% increase in turbulence under attack (p less than 0.001), validating the hypothesis of internal conflict. Conversely, Gemma-2B displays a 22.0% decrease in turbulence, characterizing a distinct, low-entropy "reflex-based" refusal mechanism. These findings demonstrate that Semantic Turbulence serves not only as a lightweight, real-time jailbreak detector but also as a non-invasive diagnostic tool for categorizing the underlying safety architecture of black-box models.

摘要: 随着大型语言模型（LLM）变得无处不在，保护它们免受对抗性“越狱”攻击的挑战也随之加剧。当前的防御策略通常依赖于计算昂贵的外部分类器或脆弱的词汇过滤器，忽视了模型推理过程的内在动态。在这项工作中，引入了层流假说，该假说假设良性输入会在LLM的多维潜在空间中引发平滑、渐进的转变，而对抗性提示会触发混乱、高方差轨迹--称为语义湍流--这是由于安全一致和遵循策略的目标之间的内部冲突。这种现象通过一种新颖的零射击度量（分层cos速度的方差）形式化。对各种小语言模型的实验评估揭示了其惊人的诊断能力。RLHF对齐的Qwen 2 -1.5B在攻击下的湍流增加了75.4%（p小于0.001），验证了内部冲突的假设。相反，Gemma-2B的湍流减少了22.0%，具有独特的低熵“基于反射”的拒绝机制。这些发现表明，Semantic Turbulence不仅可以作为一种轻量级的实时越狱检测器，而且可以作为一种非侵入性诊断工具，用于对黑匣子模型的底层安全架构进行分类。



## **25. One Leak Away: How Pretrained Model Exposure Amplifies Jailbreak Risks in Finetuned LLMs**

一个漏洞：预训练模型暴露如何放大Finetuned LLM的越狱风险 cs.CR

17 pages

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.14751v1) [paper-pdf](https://arxiv.org/pdf/2512.14751v1)

**Authors**: Yixin Tan, Zhe Yu, Jun Sakuma

**Abstract**: Finetuning pretrained large language models (LLMs) has become the standard paradigm for developing downstream applications. However, its security implications remain unclear, particularly regarding whether finetuned LLMs inherit jailbreak vulnerabilities from their pretrained sources. We investigate this question in a realistic pretrain-to-finetune threat model, where the attacker has white-box access to the pretrained LLM and only black-box access to its finetuned derivatives. Empirical analysis shows that adversarial prompts optimized on the pretrained model transfer most effectively to its finetuned variants, revealing inherited vulnerabilities from pretrained to finetuned LLMs. To further examine this inheritance, we conduct representation-level probing, which shows that transferable prompts are linearly separable within the pretrained hidden states, suggesting that universal transferability is encoded in pretrained representations. Building on this insight, we propose the Probe-Guided Projection (PGP) attack, which steers optimization toward transferability-relevant directions. Experiments across multiple LLM families and diverse finetuned tasks confirm PGP's strong transfer success, underscoring the security risks inherent in the pretrain-to-finetune paradigm.

摘要: 微调预训练的大型语言模型（LLM）已成为开发下游应用程序的标准范式。然而，其安全影响仍不清楚，特别是关于微调的LLM是否会从其预先训练的来源继承越狱漏洞。我们在现实的预训练到微调威胁模型中研究这个问题，其中攻击者可以白盒访问预训练的LLM，并且只能黑盒访问其微调衍生品。实证分析表明，在预训练模型上优化的对抗提示最有效地转移到其微调变体，揭示了从预训练到微调LLM的遗传漏洞。为了进一步检查这种继承，我们进行了表示级探测，这表明可转移提示在预训练的隐藏状态内是线性可分离的，这表明普遍可转移性被编码在预训练的表示中。基于这一见解，我们提出了探测引导投影（PGP）攻击，它将优化引导向可移植性相关方向。跨多个LLM系列和不同的微调任务的实验证实了PGP的强大转移成功，凸显了预训练到微调范式固有的安全风险。



## **26. The Role of AI in Modern Penetration Testing**

人工智能在现代渗透测试中的作用 cs.SE

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12326v1) [paper-pdf](https://arxiv.org/pdf/2512.12326v1)

**Authors**: J. Alexander Curtis, Nasir U. Eisty

**Abstract**: Penetration testing is a cornerstone of cybersecurity, traditionally driven by manual, time-intensive processes. As systems grow in complexity, there is a pressing need for more scalable and efficient testing methodologies. This systematic literature review examines how Artificial Intelligence (AI) is reshaping penetration testing, analyzing 58 peer-reviewed studies from major academic databases. Our findings reveal that while AI-assisted pentesting is still in its early stages, notable progress is underway, particularly through Reinforcement Learning (RL), which was the focus of 77% of the reviewed works. Most research centers on the discovery and exploitation phases of pentesting, where AI shows the greatest promise in automating repetitive tasks, optimizing attack strategies, and improving vulnerability identification. Real-world applications remain limited but encouraging, including the European Space Agency's PenBox and various open-source tools. These demonstrate AI's potential to streamline attack path analysis, analyze complex network topology, and reduce manual workload. However, challenges persist: current models often lack flexibility and are underdeveloped for the reconnaissance and post-exploitation phases of pentesting. Applications involving Large Language Models (LLMs) remain relatively under-researched, pointing to a promising direction for future exploration. This paper offers a critical overview of AI's current and potential role in penetration testing, providing valuable insights for researchers, practitioners, and organizations aiming to enhance security assessments through advanced automation or looking for gaps in existing research.

摘要: 渗透测试是网络安全的基石，传统上由手动、耗时的流程驱动。随着系统复杂性的增长，迫切需要更可扩展和更有效的测试方法。这篇系统性的文献综述探讨了人工智能（AI）如何重塑渗透测试，分析了来自主要学术数据库的58项同行评议研究。我们的研究结果表明，虽然人工智能辅助渗透测试仍处于早期阶段，但正在取得显着进展，特别是通过强化学习（RL），这是77%的审查工作的重点。大多数研究都集中在笔记本测试的发现和利用阶段，人工智能在自动化重复性任务、优化攻击策略和改进漏洞识别方面表现出了最大的希望。现实世界的应用程序仍然有限，但令人鼓舞，包括欧洲航天局的PenBox和各种开源工具。这些证明了人工智能在简化攻击路径分析、分析复杂网络布局和减少手动工作量方面的潜力。然而，挑战依然存在：当前的模型通常缺乏灵活性，并且不足以适应冥想的侦察和后开发阶段。涉及大型语言模型（LLM）的应用仍然相对缺乏研究，这为未来的探索指明了一个有希望的方向。本文对人工智能在渗透测试中的当前和潜在作用进行了批判性概述，为旨在通过先进自动化增强安全评估或寻找现有研究差距的研究人员、从业者和组织提供了宝贵的见解。



## **27. Taint-Based Code Slicing for LLMs-based Malicious NPM Package Detection**

基于污点的代码切片用于基于LLMs的恶意NPM包检测 cs.CR

17 pages, 4 figures, 9 tables

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12313v1) [paper-pdf](https://arxiv.org/pdf/2512.12313v1)

**Authors**: Dang-Khoa Nguyen, Gia-Thang Ho, Quang-Minh Pham, Tuyet A. Dang-Thi, Minh-Khanh Vu, Thanh-Cong Nguyen, Phat T. Tran-Truong, Duc-Ly Vu

**Abstract**: The increasing sophistication of malware attacks in the npm ecosystem, characterized by obfuscation and complex logic, necessitates advanced detection methods. Recently, researchers have turned their attention from traditional detection approaches to Large Language Models (LLMs) due to their strong capabilities in semantic code understanding. However, while LLMs offer superior semantic reasoning for code analysis, their practical application is constrained by limited context windows and high computational cost. This paper addresses this challenge by introducing a novel framework that leverages code slicing techniques for an LLM-based malicious package detection task. We propose a specialized taintbased slicing technique for npm packages, augmented by a heuristic backtracking mechanism to accurately capture malicious data flows across asynchronous, event-driven patterns (e.g., callbacks and Promises) that elude traditional analysis. An evaluation on a dataset of more than 5000 malicious and benign npm packages demonstrates that our approach isolates security-relevant code, reducing input volume by over 99% while preserving critical behavioral semantics. Using the DeepSeek-Coder-6.7B model as the classification engine, our approach achieves a detection accuracy of 87.04%, substantially outperforming a naive token-splitting baseline (75.41%) and a traditional static-analysis-based approach. These results indicate that semantically optimized input representation via code slicing not only mitigates the LLM context-window bottleneck but also significantly enhances reasoning precision for security tasks, providing an efficient and effective defense against evolving malicious open-source packages.

摘要: nPM生态系统中恶意软件攻击日益复杂，其特征是模糊和复杂的逻辑，因此需要先进的检测方法。最近，由于大型语言模型（LLM）在语义代码理解方面的强大能力，研究人员将注意力从传统的检测方法转向大型语言模型（LLM）。然而，虽然LLM为代码分析提供了卓越的语义推理，但其实际应用受到上下文窗口有限和计算成本高的限制。本文通过引入一种新颖的框架来解决这一挑战，该框架利用代码切片技术来执行基于LLM的恶意包检测任务。我们为nPM包提出了一种专门的基于污点的切片技术，并通过启发式回溯机制进行增强，以准确地捕获跨同步、事件驱动模式的恶意数据流（例如，回调和承诺）无法实现传统分析。对5000多个恶意和良性nPM包的数据集的评估表明，我们的方法隔离了安全相关代码，将输入量减少了99%以上，同时保留了关键行为语义。使用DeepSeek-Coder-6.7B模型作为分类引擎，我们的方法实现了87.04%的检测准确率，大大优于原始符号分裂基线（75.41%）和传统的基于静态分析的方法。这些结果表明，通过代码切片进行语义优化的输入表示不仅缓解了LLM上下文窗口瓶颈，而且显着提高了安全任务的推理精度，为不断发展的恶意开源包提供了高效且有效的防御。



## **28. Keep the Lights On, Keep the Lengths in Check: Plug-In Adversarial Detection for Time-Series LLMs in Energy Forecasting**

保持灯亮着，保持警惕：能源预测中时间序列LLM的插入式对抗检测 cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12154v1) [paper-pdf](https://arxiv.org/pdf/2512.12154v1)

**Authors**: Hua Ma, Ruoxi Sun, Minhui Xue, Xingliang Yuan, Carsten Rudolph, Surya Nepal, Ling Liu

**Abstract**: Accurate time-series forecasting is increasingly critical for planning and operations in low-carbon power systems. Emerging time-series large language models (TS-LLMs) now deliver this capability at scale, requiring no task-specific retraining, and are quickly becoming essential components within the Internet-of-Energy (IoE) ecosystem. However, their real-world deployment is complicated by a critical vulnerability: adversarial examples (AEs). Detecting these AEs is challenging because (i) adversarial perturbations are optimized across the entire input sequence and exploit global temporal dependencies, which renders local detection methods ineffective, and (ii) unlike traditional forecasting models with fixed input dimensions, TS-LLMs accept sequences of variable length, increasing variability that complicates detection. To address these challenges, we propose a plug-in detection framework that capitalizes on the TS-LLM's own variable-length input capability. Our method uses sampling-induced divergence as a detection signal. Given an input sequence, we generate multiple shortened variants and detect AEs by measuring the consistency of their forecasts: Benign sequences tend to produce stable predictions under sampling, whereas adversarial sequences show low forecast similarity, because perturbations optimized for a full-length sequence do not transfer reliably to shorter, differently-structured subsamples. We evaluate our approach on three representative TS-LLMs (TimeGPT, TimesFM, and TimeLLM) across three energy datasets: ETTh2 (Electricity Transformer Temperature), NI (Hourly Energy Consumption), and Consumption (Hourly Electricity Consumption and Production). Empirical results confirm strong and robust detection performance across both black-box and white-box attack scenarios, highlighting its practicality as a reliable safeguard for TS-LLM forecasting in real-world energy systems.

摘要: 准确的时间序列预测对于低碳电力系统的规划和运营越来越重要。新兴的时间序列大型语言模型（TS-LLM）现在大规模提供了这一能力，无需针对特定任务的再培训，并且正在迅速成为能源互联网（IoE）生态系统中的重要组成部分。然而，它们的现实世界部署因一个关键漏洞而变得复杂：对抗性示例（AE）。检测这些AE具有挑战性，因为（i）对抗性扰动在整个输入序列中得到优化并利用全局时间依赖性，这使得局部检测方法无效，并且（ii）与具有固定输入维度的传统预测模型不同，TS-LLM接受可变长度的序列，增加了使检测复杂化的可变性。为了应对这些挑战，我们提出了一种插件检测框架，该框架利用了TS-LLM自身的可变长度输入能力。我们的方法使用采样引起的分歧作为检测信号。给定一个输入序列，我们生成多个缩短的变体，并通过测量其预测的一致性来检测AE：良性序列往往会在抽样下产生稳定的预测，而对抗序列显示出较低的预测相似性，因为针对全长序列优化的扰动不会可靠地转移到更短、结构不同的子样本。我们在三个能源数据集的三个代表性TS-LLM（TimeGPT、TimesFM和TimeLLM）上评估了我们的方法：ETTh 2（Transformer温度）、NI（小时能源消耗）和消耗（小时电力消耗和生产）。经验结果证实了在黑匣子和白盒攻击场景中强大且稳健的检测性能，凸显了其作为现实世界能源系统中TS-LLM预测的可靠保障的实用性。



## **29. BRIDG-ICS: AI-Grounded Knowledge Graphs for Intelligent Threat Analytics in Industry~5.0 Cyber-Physical Systems**

BRIDG-ICS：工业~5.0网络物理系统中智能威胁分析的基于人工智能的知识图 cs.CR

44 Pages, To be published in Springer Cybersecurity Journal

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12112v1) [paper-pdf](https://arxiv.org/pdf/2512.12112v1)

**Authors**: Padmeswari Nandiya, Ahmad Mohsin, Ahmed Ibrahim, Iqbal H. Sarker, Helge Janicke

**Abstract**: Industry 5.0's increasing integration of IT and OT systems is transforming industrial operations but also expanding the cyber-physical attack surface. Industrial Control Systems (ICS) face escalating security challenges as traditional siloed defences fail to provide coherent, cross-domain threat insights. We present BRIDG-ICS (BRIDge for Industrial Control Systems), an AI-driven Knowledge Graph (KG) framework for context-aware threat analysis and quantitative assessment of cyber resilience in smart manufacturing environments. BRIDG-ICS fuses heterogeneous industrial and cybersecurity data into an integrated Industrial Security Knowledge Graph linking assets, vulnerabilities, and adversarial behaviours with probabilistic risk metrics (e.g. exploit likelihood, attack cost). This unified graph representation enables multi-stage attack path simulation using graph-analytic techniques. To enrich the graph's semantic depth, the framework leverages Large Language Models (LLMs): domain-specific LLMs extract cybersecurity entities, predict relationships, and translate natural-language threat descriptions into structured graph triples, thereby populating the knowledge graph with missing associations and latent risk indicators. This unified AI-enriched KG supports multi-hop, causality-aware threat reasoning, improving visibility into complex attack chains and guiding data-driven mitigation. In simulated industrial scenarios, BRIDG-ICS scales well, reduces potential attack exposure, and can enhance cyber-physical system resilience in Industry 5.0 settings.

摘要: 工业5.0对IT和OT系统的日益整合正在改变工业运营，但也扩大了网络物理攻击面。工业控制系统（ICS）面临着不断升级的安全挑战，因为传统的孤立防御无法提供连贯的跨域威胁洞察。我们提出了BRIDG-ICS（BRIDge for Industrial Control Systems），这是一个人工智能驱动的知识图（KG）框架，用于智能制造环境中的上下文感知威胁分析和网络弹性的定量评估。BRIDG-ICS将异构的工业和网络安全数据融合到一个集成的工业安全知识图中，将资产、漏洞和对抗行为与概率风险指标（例如，利用可能性、攻击成本）联系起来。这种统一的图形表示可以使用图形分析技术进行多阶段攻击路径模拟。为了丰富图形的语义深度，该框架利用大型语言模型（LLM）：特定领域的LLM提取网络安全实体、预测关系，并将自然语言威胁描述翻译为结构化图形三重体，从而用缺失的关联和潜在风险指标填充知识图形。这款统一的、富含人工智能的KG支持多跳、疏忽感知的威胁推理，提高对复杂攻击链的可见性并指导数据驱动的缓解。在模拟工业场景中，BRIDG-ICS可扩展性良好，减少了潜在的攻击暴露，并可以增强工业5.0环境中的网络物理系统弹性。



## **30. Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring**

利用代表性对比评分重新思考大视觉语言模型的越狱检测 cs.CR

40 pages, 13 figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.12069v1) [paper-pdf](https://arxiv.org/pdf/2512.12069v1)

**Authors**: Peichun Hua, Hao Li, Shanghao Shi, Zhiyuan Yu, Ning Zhang

**Abstract**: Large Vision-Language Models (LVLMs) are vulnerable to a growing array of multimodal jailbreak attacks, necessitating defenses that are both generalizable to novel threats and efficient for practical deployment. Many current strategies fall short, either targeting specific attack patterns, which limits generalization, or imposing high computational overhead. While lightweight anomaly-detection methods offer a promising direction, we find that their common one-class design tends to confuse novel benign inputs with malicious ones, leading to unreliable over-rejection. To address this, we propose Representational Contrastive Scoring (RCS), a framework built on a key insight: the most potent safety signals reside within the LVLM's own internal representations. Our approach inspects the internal geometry of these representations, learning a lightweight projection to maximally separate benign and malicious inputs in safety-critical layers. This enables a simple yet powerful contrastive score that differentiates true malicious intent from mere novelty. Our instantiations, MCD (Mahalanobis Contrastive Detection) and KCD (K-nearest Contrastive Detection), achieve state-of-the-art performance on a challenging evaluation protocol designed to test generalization to unseen attack types. This work demonstrates that effective jailbreak detection can be achieved by applying simple, interpretable statistical methods to the appropriate internal representations, offering a practical path towards safer LVLM deployment. Our code is available on Github https://github.com/sarendis56/Jailbreak_Detection_RCS.

摘要: 大型视觉语言模型（LVLM）容易受到越来越多的多模式越狱攻击的影响，因此需要既可推广到新型威胁又可有效实际部署的防御。当前的许多策略都存在缺陷，要么针对特定的攻击模式（这限制了概括性），要么施加了很高的计算负担。虽然轻量级异常检测方法提供了一个有希望的方向，但我们发现它们常见的一类设计往往会混淆新颖的良性输入与恶意输入，从而导致不可靠的过度拒绝。为了解决这个问题，我们提出了代表性对比评分（RC），这是一个建立在关键见解之上的框架：最有力的安全信号位于LVLM自己的内部表示中。我们的方法检查这些表示的内部几何形状，学习轻量级投影以最大限度地分离安全关键层中的良性和恶意输入。这使得可以获得一个简单而强大的对比分数，将真正的恶意意图与纯粹的新颖性区分开来。我们的实例BCD（Mahalanobis Contrasive Detection）和KCD（K-nearest Contrasive Detection）在具有挑战性的评估协议上实现了最先进的性能，该协议旨在测试对未见攻击类型的概括。这项工作表明，通过将简单、可解释的统计方法应用于适当的内部表示，可以实现有效的越狱检测，从而提供了实现更安全LVLM部署的实用途径。我们的代码可在Github https://github.com/sarendis56/Jailbreak_Detection_RCS上获取。



## **31. Learning to Extract Context for Context-Aware LLM Inference**

学习为上下文感知LLM推理提取上下文 cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11986v1) [paper-pdf](https://arxiv.org/pdf/2512.11986v1)

**Authors**: Minseon Kim, Lucas Caccia, Zhengyan Shi, Matheus Pereira, Marc-Alexandre Côté, Xingdi Yuan, Alessandro Sordoni

**Abstract**: User prompts to large language models (LLMs) are often ambiguous or under-specified, and subtle contextual cues shaped by user intentions, prior knowledge, and risk factors strongly influence what constitutes an appropriate response. Misinterpreting intent or risks may lead to unsafe outputs, while overly cautious interpretations can cause unnecessary refusal of benign requests. In this paper, we question the conventional framework in which LLMs generate immediate responses to requests without considering broader contextual factors. User requests are situated within broader contexts such as intentions, knowledge, and prior experience, which strongly influence what constitutes an appropriate answer. We propose a framework that extracts and leverages such contextual information from the user prompt itself. Specifically, a reinforcement learning based context generator, designed in an autoencoder-like fashion, is trained to infer contextual signals grounded in the prompt and use them to guide response generation. This approach is particularly important for safety tasks, where ambiguous requests may bypass safeguards while benign but confusing requests can trigger unnecessary refusals. Experiments show that our method reduces harmful responses by an average of 5.6% on the SafetyInstruct dataset across multiple foundation models and improves the harmonic mean of attack success rate and compliance on benign prompts by 6.2% on XSTest and WildJailbreak. These results demonstrate the effectiveness of context extraction for safer and more reliable LLM inferences.

摘要: 用户对大型语言模型（LLM）的提示通常是模糊的或未指定的，由用户意图、先验知识和风险因素塑造的微妙上下文线索强烈影响适当响应的构成。误解意图或风险可能会导致不安全的输出，而过于谨慎的解释可能会导致对善意请求的不必要的拒绝。在本文中，我们质疑传统框架，在该框架中，LLM在不考虑更广泛的背景因素的情况下立即对请求做出响应。用户请求位于更广泛的背景下，例如意图、知识和先前的经验，这些都强烈影响合适的答案的构成。我们提出了一个框架，可以从用户提示本身提取和利用此类上下文信息。具体来说，以类似自动编码器的方式设计的基于强化学习的上下文生成器被训练为推断基于提示的上下文信号，并使用它们来指导响应生成。这种方法对于安全任务尤其重要，其中模棱两可的请求可能会绕过保障措施，而善意但令人困惑的请求可能会引发不必要的拒绝。实验表明，我们的方法在多个基础模型的SafetyDirect数据集中平均减少了5.6%，并在XSTest和WildJailbreak上将良性提示的攻击成功率和合规性的调和平均值提高了6.2%。这些结果证明了上下文提取的有效性，更安全，更可靠的LLM推理。



## **32. Super Suffixes: Bypassing Text Generation Alignment and Guard Models Simultaneously**

超级后缀：同时简化文本生成对齐和保护模型 cs.CR

13 pages, 5 Figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11783v1) [paper-pdf](https://arxiv.org/pdf/2512.11783v1)

**Authors**: Andrew Adiletta, Kathryn Adiletta, Kemal Derya, Berk Sunar

**Abstract**: The rapid deployment of Large Language Models (LLMs) has created an urgent need for enhanced security and privacy measures in Machine Learning (ML). LLMs are increasingly being used to process untrusted text inputs and even generate executable code, often while having access to sensitive system controls. To address these security concerns, several companies have introduced guard models, which are smaller, specialized models designed to protect text generation models from adversarial or malicious inputs. In this work, we advance the study of adversarial inputs by introducing Super Suffixes, suffixes capable of overriding multiple alignment objectives across various models with different tokenization schemes. We demonstrate their effectiveness, along with our joint optimization technique, by successfully bypassing the protection mechanisms of Llama Prompt Guard 2 on five different text generation models for malicious text and code generation. To the best of our knowledge, this is the first work to reveal that Llama Prompt Guard 2 can be compromised through joint optimization.   Additionally, by analyzing the changing similarity of a model's internal state to specific concept directions during token sequence processing, we propose an effective and lightweight method to detect Super Suffix attacks. We show that the cosine similarity between the residual stream and certain concept directions serves as a distinctive fingerprint of model intent. Our proposed countermeasure, DeltaGuard, significantly improves the detection of malicious prompts generated through Super Suffixes. It increases the non-benign classification rate to nearly 100%, making DeltaGuard a valuable addition to the guard model stack and enhancing robustness against adversarial prompt attacks.

摘要: 大型语言模型（LLM）的快速部署迫切需要在机器学习（ML）中增强安全和隐私措施。LLM越来越多地被用于处理不受信任的文本输入，甚至生成可执行代码，通常是在可以访问敏感系统控制的情况下。为了解决这些安全问题，几家公司引入了防护模型，这是一种更小的专门模型，旨在保护文本生成模型免受对抗性或恶意输入的影响。在这项工作中，我们通过引入超级后缀来推进对抗性输入的研究，超级后缀能够覆盖具有不同标记化方案的各种模型中的多个对齐目标。我们通过在恶意文本和代码生成的五种不同文本生成模型上成功绕过Llama Promise Guard 2的保护机制，证明了它们以及我们的联合优化技术的有效性。据我们所知，这是第一部揭示Llama Promise Guard 2可以通过联合优化而受到损害的作品。   此外，通过分析令牌序列处理期间模型内部状态与特定概念方向的相似性变化，我们提出了一种有效且轻量级的方法来检测超级后缀攻击。我们表明，剩余流和某些概念方向之间的cos相似性可以作为模型意图的独特指纹。我们提出的对策Delta Guard显着改进了对通过超级后缀生成的恶意提示的检测。它将非良性分类率提高到近100%，使Delta Guard成为保护模型堆栈的宝贵补充，并增强了针对对抗提示攻击的鲁棒性。



## **33. When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection**

当批评变成接受：量化基于LLM-based科学评论员间接提示注入的脆弱性 cs.AI

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.10449v2) [paper-pdf](https://arxiv.org/pdf/2512.10449v2)

**Authors**: Devanshu Sahoo, Manish Prasad, Vasudev Majhi, Jahnvi Singh, Vinay Chamola, Yash Sinha, Murari Mandal, Dhruv Kumar

**Abstract**: The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic.

摘要: 随着大型语言模型（LLM）的集成，科学同行评审的格局正在迅速发展。这种转变是由两个平行趋势推动的：评审员广泛采用法学硕士来管理工作量（“懒惰评审员”假设），以及AAAI和斯坦福大学Agents 4Science等会议正式机构部署人工智能驱动的评估系统。本研究调查了这些“法学硕士作为法官”系统（包括非法的和受制裁的）对对抗性PDF操纵的稳健性。与一般的越狱不同，我们专注于一个独特的激励：将“卸载”决策转换为“接受”，为此我们开发了一种新型的评估指标，称为WAVS（加权对抗性脆弱性分数）。我们策划了一个包含200篇科学论文的数据集，并为这项任务调整了15种特定领域的攻击策略，在13种语言模型中进行了评估，包括GPT-5，Claude Haiku和DeepSeek。我们的研究结果表明，混淆策略，如“最大马克Magyk”成功地操纵分数，即使在大规模的模型中也能达到惊人的决策翻转率。我们将发布完整的数据集和注入框架，以促进对该主题的更多研究。



## **34. Phishing Email Detection Using Large Language Models**

使用大型语言模型的网络钓鱼电子邮件检测 cs.CR

7 pages

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.10104v2) [paper-pdf](https://arxiv.org/pdf/2512.10104v2)

**Authors**: Najmul Hasan, Prashanth BusiReddyGari, Haitao Zhao, Yihao Ren, Jinsheng Xu, Shaohu Zhang

**Abstract**: Email phishing is one of the most prevalent and globally consequential vectors of cyber intrusion. As systems increasingly deploy Large Language Models (LLMs) applications, these systems face evolving phishing email threats that exploit their fundamental architectures. Current LLMs require substantial hardening before deployment in email security systems, particularly against coordinated multi-vector attacks that exploit architectural vulnerabilities. This paper proposes LLMPEA, an LLM-based framework to detect phishing email attacks across multiple attack vectors, including prompt injection, text refinement, and multilingual attacks. We evaluate three frontier LLMs (e.g., GPT-4o, Claude Sonnet 4, and Grok-3) and comprehensive prompting design to assess their feasibility, robustness, and limitations against phishing email attacks. Our empirical analysis reveals that LLMs can detect the phishing email over 90% accuracy while we also highlight that LLM-based phishing email detection systems could be exploited by adversarial attack, prompt injection, and multilingual attacks. Our findings provide critical insights for LLM-based phishing detection in real-world settings where attackers exploit multiple vulnerabilities in combination.

摘要: 电子邮件网络钓鱼是最普遍、最具全球影响力的网络入侵载体之一。随着系统越来越多地部署大型语言模型（LLM）应用程序，这些系统面临着利用其基本架构的不断发展的网络钓鱼电子邮件威胁。当前的LLM在部署到电子邮件安全系统之前需要进行实质性的强化，特别是针对利用架构漏洞的协调多载体攻击。本文提出了LLMPEA，这是一个基于LLM的框架，用于检测跨多种攻击载体的网络钓鱼电子邮件攻击，包括提示注入、文本细化和多语言攻击。我们评估了三个前沿LLM（例如，GPT-4 o、Claude Sonnet 4和Grok-3）以及全面的提示设计，以评估其可行性、稳健性和针对网络钓鱼电子邮件攻击的限制。我们的实证分析表明，LLM可以检测到超过90%的网络钓鱼电子邮件，同时我们还强调，基于LLM的网络钓鱼电子邮件检测系统可能会被对抗性攻击、提示注入和多语言攻击所利用。我们的研究结果为攻击者组合利用多个漏洞的现实环境中基于LLM的网络钓鱼检测提供了重要见解。



## **35. CNFinBench: A Benchmark for Safety and Compliance of Large Language Models in Finance**

CNFinBench：金融领域大型语言模型安全和合规性的基准 cs.CE

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.09506v2) [paper-pdf](https://arxiv.org/pdf/2512.09506v2)

**Authors**: Jinru Ding, Chao Ding, Wenrao Pang, Boyi Xiao, Zhiqiang Liu, Pengcheng Chen, Jiayuan Chen, Tiantian Yuan, Junming Guan, Yidong Jiang, Dawei Cheng, Jie Xu

**Abstract**: Large language models (LLMs) are increasingly deployed across the financial sector for tasks like investment research and algorithmic trading. Their high-stakes nature demands rigorous evaluation of models' safety and regulatory alignment. However, there is a significant gap between evaluation capabilities and safety requirements. Current financial benchmarks mainly focus on textbook-style question answering and numerical problem-solving, failing to simulate the open-ended scenarios where safety risks typically manifest. To close these gaps, we introduce CNFinBench, a benchmark structured around a Capability-Compliance-Safety triad encompassing 15 subtasks. For Capability Q&As, we introduce a novel business-vertical taxonomy aligned with core financial domains like banking operations, which allows institutions to assess model readiness for deployment in operational scenarios. For Compliance and Risk Control Q&As, we embed regulatory requirements within realistic business scenarios to ensure models are evaluated under practical, scenario-driven conditions. For Safety Q&As, we uniquely incorporate structured bias and fairness auditing, a dimension overlooked by other holistic financial benchmarks, and introduce the first multi-turn adversarial dialogue task to systematically expose compliance decay under sustained, context-aware attacks. Accordingly, we propose the Harmful Instruction Compliance Score (HICS) to quantify models' consistency in resisting harmful instructions across multi-turn dialogues. Experiments on 21 models across all subtasks reveal a persistent gap between capability and compliance: models achieve an average score of 61.0 on capability tasks but drop to 34.2 on compliance and risk-control evaluations. In multi-turn adversarial dialogue tests, most LLMs attain only partial resistance, demonstrating that refusal alone is insufficient without cited, verifiable reasoning.

摘要: 大型语言模型（LLM）越来越多地被部署在金融领域，用于投资研究和算法交易等任务。它们的高风险性质要求对模型的安全性和监管一致性进行严格评估。然而，评估能力与安全要求之间存在明显差距。当前的财务基准主要集中在教科书式的问答和数字问题解决上，未能模拟安全风险通常显现的开放式场景。为了缩小这些差距，我们引入了CNFinBench，这是一个围绕能力-合规-安全三位一体构建的基准，包含15个子任务。对于能力问答，我们引入了一种与银行运营等核心金融领域保持一致的新型业务垂直分类法，使机构能够评估模型在运营场景中部署的准备情况。对于合规和风险控制问答，我们将监管要求嵌入现实的业务场景中，以确保模型在实际的、业务驱动的条件下进行评估。对于安全问答，我们独特地结合了结构性偏见和公平性审计（这是其他整体财务基准所忽视的一个维度），并引入了第一个多回合对抗性对话任务，以系统性地揭露持续、上下文感知攻击下的合规性衰退。因此，我们提出了有害指令合规评分（HICS）来量化模型在多轮对话中抵抗有害指令的一致性。对所有子任务的21个模型进行的实验揭示了能力与合规性之间持续存在的差距：模型在能力任务上的平均得分为61.0，但在合规和风险控制评估上的平均得分下降至34.2。在多轮对抗性对话测试中，大多数LLM仅获得部分抵抗，这表明如果没有引用的、可验证的推理，仅靠拒绝是不够的。



## **36. The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search**

木马知识：通过无害提示编织和自适应树搜索破解商业LLM护栏 cs.CR

Updated with new baselines and experimental results

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.01353v3) [paper-pdf](https://arxiv.org/pdf/2512.01353v3)

**Authors**: Rongzhe Wei, Peizhi Niu, Xinjie Shen, Tony Tu, Yifan Li, Ruihan Wu, Eli Chien, Pin-Yu Chen, Olgica Milenkovic, Pan Li

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Existing approaches overwhelmingly operate within the prompt-optimization paradigm: whether through traditional algorithmic search or recent agent-based workflows, the resulting prompts typically retain malicious semantic signals that modern guardrails are primed to detect. In contrast, we identify a deeper, largely overlooked vulnerability stemming from the highly interconnected nature of an LLM's internal knowledge. This structure allows harmful objectives to be realized by weaving together sequences of benign sub-queries, each of which individually evades detection. To exploit this loophole, we introduce the Correlated Knowledge Attack Agent (CKA-Agent), a dynamic framework that reframes jailbreaking as an adaptive, tree-structured exploration of the target model's knowledge base. The CKA-Agent issues locally innocuous queries, uses model responses to guide exploration across multiple paths, and ultimately assembles the aggregated information to achieve the original harmful objective. Evaluated across state-of-the-art commercial LLMs (Gemini2.5-Flash/Pro, GPT-oss-120B, Claude-Haiku-4.5), CKA-Agent consistently achieves over 95% success rates even against strong guardrails, underscoring the severity of this vulnerability and the urgent need for defenses against such knowledge-decomposition attacks. Our codes are available at https://github.com/Graph-COM/CKA-Agent.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，这些攻击绕过安全护栏以引发有害输出。现有方法绝大多数在预算优化范式下运行：无论是通过传统的算法搜索还是最近的基于代理的工作流程，产生的提示通常都会保留现代护栏准备好检测的恶意语义信号。相比之下，我们发现了一个更深层次的、在很大程度上被忽视的漏洞，该漏洞源于法学硕士内部知识的高度相互关联的性质。这种结构允许通过将良性子查询序列编织在一起来实现有害目标，每个子查询都单独逃避检测。为了利用这个漏洞，我们引入了相关知识攻击代理（CKA-Agent），这是一个动态框架，它将越狱重新构建为对目标模型知识库的自适应、树结构化探索。CKA-Agent发出本地无害的查询，使用模型响应来指导跨多个路径的探索，并最终聚集信息以实现最初的有害目标。经过最先进的商业LLM（Gemini 2.5-Flash/Pro、GPT-oss-120 B、Claude-Haiku-4.5）的评估，即使在强大的护栏下，CKA-Agent也始终实现了超过95%的成功率，凸显了该漏洞的严重性以及对此类知识分解攻击的防御的迫切需要。我们的代码可在https://github.com/Graph-COM/CKA-Agent上获取。



## **37. Phantom Menace: Exploring and Enhancing the Robustness of VLA Models Against Physical Sensor Attacks**

幻影威胁：探索和增强VLA模型对抗物理传感器攻击的鲁棒性 cs.RO

Accepted by AAAI 2026 main track

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2511.10008v2) [paper-pdf](https://arxiv.org/pdf/2511.10008v2)

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored. To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel "Real-Sim-Real" framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.

摘要: 视觉-语言-动作（VLA）模型通过实现端到端的感知到动作管道，彻底改变了机器人系统，该管道集成了多种感官模式，例如由摄像机处理的视觉信号和由麦克风捕获的听觉信号。这种多模式集成使VLA模型能够使用不同的传感器数据流来解释复杂的现实世界环境。鉴于基于VLA的系统严重依赖感官输入，VLA模型对抗物理世界传感器攻击的安全性仍然严重不足。为了弥补这一差距，我们首次对针对VLA的物理传感器攻击进行了系统研究，量化了传感器攻击的影响并调查VLA模型的防御。我们引入了一个新颖的“Real-Sim-Real”框架，该框架自动模拟基于物理的传感器攻击载体，包括六次针对摄像头和两个针对麦克风的攻击，并在真实的机器人系统上对其进行验证。通过在不同攻击参数下对各种VLA架构和任务进行大规模评估，我们展示了显着的漏洞，其易感性模式揭示了对任务类型和模型设计的关键依赖性。我们进一步开发了一种基于对抗训练的防御，可以增强VLA对传感器攻击引起的分布外物理扰动的鲁棒性，同时保持模型性能。我们的研究结果揭示了迫切需要标准化的稳健性基准和缓解策略，以确保VLA在安全关键环境中的部署。



## **38. Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models**

对生成性基因组模型的生物知情混合成员推断攻击 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2511.07503v3) [paper-pdf](https://arxiv.org/pdf/2511.07503v3)

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs.

摘要: 遗传数据可用性的增加改变了基因组学研究，但由于其敏感性，对其处理提出了许多隐私问题。这项工作探索了使用语言模型（LM）来生成合成基因突变谱，利用差异隐私（DP）来保护敏感遗传数据。我们通过引入一种新型的生物知情混合成员推断攻击（biHMIA）来经验性地评估DP模式的隐私保证，该攻击将传统的黑匣子MIA与上下文基因组学指标相结合，以增强攻击能力。我们的实验表明，小型和大型Transformer GPT类模型都是小规模基因组学的可行合成变体生成器，并且与传统的基于度量的MIA相比，我们的混合攻击平均会导致更高的对抗成功。



## **39. RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines**

RAGrank：使用PageRank来应对RTI LLM管道中的中毒 cs.CR

Presented as a poster at the Annual Computer Security Applications Conference (ACSAC) 2025

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2510.20768v2) [paper-pdf](https://arxiv.org/pdf/2510.20768v2)

**Authors**: Austin Jia, Avaneesh Ramesh, Zain Shamsi, Daniel Zhang, Alex Liu

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds.

摘要: 检索增强生成（RAG）已成为在网络威胁情报（RTI）系统中操作大型语言模型（LLM）使用的主要架构模式。然而，这种设计很容易受到中毒攻击，并且之前提出的防御措施可能会在RTI上下文中失败，因为网络威胁信息对于新兴攻击来说通常是全新的，而且复杂的威胁行为者可以模仿合法的格式、术语和文体惯例。为了解决这个问题，我们建议可以通过在数据库上应用源可信度算法来加速现代RAG防御的鲁棒性，以PageRank为例。在我们的实验中，我们量化地证明，我们的算法使用标准化的MS MARCO数据集，在推广受信任内容的同时，对恶意文档应用较低的权威分数。我们还在RTI文档和提要上展示了我们算法的概念验证性能。



## **40. BreakFun: Jailbreaking LLMs via Schema Exploitation**

BreakFun：通过模式利用越狱LLM cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2510.17904v2) [paper-pdf](https://arxiv.org/pdf/2510.17904v2)

**Authors**: Amirkia Rafiei Oskooei, Mehmet S. Aktas

**Abstract**: The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.

摘要: 大型语言模型（LLM）在处理结构化数据和遵守语法规则方面的熟练程度是推动其广泛采用的一种能力，但也使它们变得脆弱。在本文中，我们通过BreakFun调查了这个漏洞，BreakFun是一种越狱方法，可以将LLM对结构化模式的遵守武器化。BreakFun采用了一个由三部分组成的提示，将一个无辜的框架和一个思想链分散注意力与一个核心“特洛伊模式”相结合-一个精心制作的数据结构，迫使模型生成有害内容，利用LLM遵循结构和模式的强烈倾向。我们证明该漏洞具有高度可转移性，JailbreakBench上的13个基础和专有模型平均成功率为89%，并在几个著名模型上达到100%的攻击成功率（ASB）。一项严格的消融研究证实，该特洛伊模式是攻击的主要原因。为了解决这个问题，我们引入了对抗性提示解构护栏，这是一种利用二级LLM来执行“文字转录”的防御--提取所有人类可读的文本以隔离和揭示用户真正的有害意图。我们的概念验证护栏展示了针对攻击的高功效，验证了针对欺骗性模式是一种可行的缓解策略。我们的工作探讨了法学硕士的核心优势如何转化为关键弱点，为构建更稳健一致的模型提供了新的视角。



## **41. Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration**

Lexo：通过LLM辅助程序再生消除隐形供应链攻击 cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2510.14522v3) [paper-pdf](https://arxiv.org/pdf/2510.14522v3)

**Authors**: Evangelos Lamprou, Julian Dai, Grigoris Ntousakis, Martin C. Rinard, Nikos Vasilakis

**Abstract**: Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. An evaluation on 100+ real-world packages, including high-profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<30m on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when given the source code of the component and prompted to do so.

摘要: 软件供应链攻击是开源软件生态系统中一个重要且持续存在的问题。这些攻击保留了组件实现的标准功能，但还隐藏了仅在组件到达其目标环境时激活的恶意功能。Lexo通过自动学习和重新生成潜在恶意组件的无可识别性版本来解决此类隐形攻击。Lexo首先生成一组输入-输出对来建模组件的完整可观察行为，然后使用其合成原始组件的新版本。新组件实现了原始功能，但避免了隐蔽的恶意行为。在整个重建过程中，Lexo会咨询大型语言模型（LLM）的几个不同实例，使用正确性和覆盖率指标来引导这些实例，并保护它们的结果。对100多个现实世界的包（包括备受瞩目的隐形供应链攻击）的评估表明，Lexo可以跨多个领域扩展，有效地再生代码（平均<3000万），保持兼容性，并成功消除了几次现实世界供应链攻击中的恶意代码，即使在最先进的LLM在给定组件的源代码并提示这样做时未能消除恶意代码的情况下。



## **42. A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks**

针对即时注入攻击的多代理LLM防御管道 cs.CR

Accepted at the 11th IEEE WIECON-ECE 2025

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2509.14285v4) [paper-pdf](https://arxiv.org/pdf/2509.14285v4)

**Authors**: S M Asif Hossain, Ruksat Khan Shayoni, Mohd Ruhul Ameen, Akif Islam, M. F. Mridha, Jungpil Shin

**Abstract**: Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.

摘要: 提示注入攻击是大型语言模型（LLM）部署中的一个主要漏洞，用户输入中嵌入的恶意指令可以覆盖系统提示并引发意外行为。本文提出了一种新型的多代理防御框架，该框架在协调管道中使用专门的LLM代理来实时检测和抵消即时注入攻击。我们使用两种不同的架构来评估我们的方法：顺序代理链管道和基于分层协调器的系统。我们对两个LLM平台（ChatGLM和Llama 2）上的55种独特的即时注入攻击（分为8类，总共400个攻击实例）进行了全面评估，展示了显着的安全改进。在没有防御机制的情况下，ChatGLM的基线攻击成功率（ASB）达到30%，Llama 2的基线攻击成功率（ASB）达到20%。我们的多代理管道实现了100%的缓解，在所有测试场景中将ASB降低至0%。该框架展示了多种攻击类别的稳健性，包括直接覆盖、代码执行尝试、数据溢出和模糊技术，同时维护合法查询的系统功能。



## **43. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

相信我，我知道这个功能：使用偏差劫持LLM静态分析 cs.LG

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2508.17361v2) [paper-pdf](https://arxiv.org/pdf/2508.17361v2)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective against basic and reasoning models, but are also transferable across model families (OpenAI, Anthropic, Google), and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.

摘要: 大型语言模型（LLM）越来越被信任大规模执行自动代码审查和静态分析，支持漏洞检测、总结和重构等任务。在本文中，我们识别并利用基于LLM的代码分析中的一个关键漏洞：一种抽象偏见，导致模型过度概括熟悉的编程模式并忽略小而有意义的错误。对手可以利用这个盲点来劫持LLM解释的控制流，只需最少的编辑，并且不会影响实际的运行时行为。我们将这种攻击称为熟悉模式攻击（FTA）。   我们开发了一种全自动的黑匣子算法，可以发现并将FPA注入目标代码。我们的评估表明，PFA不仅对基本模型和推理模型有效，而且还可以跨模型家族（OpenAI、Anthropic、Google）移植，并且跨编程语言（Python、C、Rust、Go）通用。此外，即使通过强大的系统提示明确警告模型有关攻击，FPA仍然有效。最后，我们探讨了PFA的积极、防御性用途，并讨论了它们对面向代码的LLM的可靠性和安全性的更广泛影响。



## **44. ConceptGuard: Neuro-Symbolic Safety Guardrails via Sparse Interpretable Jailbreak Concepts**

ConceptGuard：通过稀疏可解释越狱概念的神经符号安全护栏 cs.CL

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2508.16325v2) [paper-pdf](https://arxiv.org/pdf/2508.16325v2)

**Authors**: Darpan Aswal, Céline Hudelot

**Abstract**: Large Language Models have found success in a variety of applications. However, their safety remains a concern due to the existence of various jailbreaking methods. Despite significant efforts, alignment and safety fine-tuning only provide a certain degree of robustness against jailbreak attacks that covertly mislead LLMs towards the generation of harmful content. This leaves them prone to a range of vulnerabilities, including targeted misuse and accidental user profiling. This work introduces \textbf{ConceptGuard}, a novel framework that leverages Sparse Autoencoders (SAEs) to identify interpretable concepts within LLM internals associated with different jailbreak themes. By extracting semantically meaningful internal representations, ConceptGuard enables building robust safety guardrails -- offering fully explainable and generalizable defenses without sacrificing model capabilities or requiring further fine-tuning. Leveraging advances in the mechanistic interpretability of LLMs, our approach provides evidence for a shared activation geometry for jailbreak attacks in the representation space, a potential foundation for designing more interpretable and generalizable safeguards against attackers.

摘要: 大型语言模型在各种应用中取得了成功。然而，由于各种越狱方法的存在，他们的安全仍然是一个问题。尽管做出了巨大的努力，但对齐和安全微调只能对越狱攻击提供一定程度的鲁棒性，这些攻击秘密误导LLM产生有害内容。这使得它们容易受到一系列漏洞，包括有针对性的滥用和意外的用户分析。这项工作引入了\textBF{ConceptGuard}，这是一个新颖的框架，它利用稀疏自动编码器（SAEs）来识别与不同越狱主题相关的LLM内部中的可解释概念。通过提取具有语义意义的内部表示，ConceptGuard能够构建强大的安全护栏--在不牺牲模型能力或需要进一步微调的情况下提供完全可解释和可概括的防御。利用LLM机械可解释性的进步，我们的方法为表示空间中越狱攻击的共享激活几何提供了证据，这是设计针对攻击者的更具可解释性和可概括性的防护措施的潜在基础。



## **45. May I have your Attention? Breaking Fine-Tuning based Prompt Injection Defenses using Architecture-Aware Attacks**

请注意吗？使用架构感知攻击突破基于微调的提示注入防御 cs.CR

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2507.07417v2) [paper-pdf](https://arxiv.org/pdf/2507.07417v2)

**Authors**: Nishit V. Pandya, Andrey Labunets, Sicun Gao, Earlence Fernandes

**Abstract**: A popular class of defenses against prompt injection attacks on large language models (LLMs) relies on fine-tuning to separate instructions and data, so that the LLM does not follow instructions that might be present with data. We evaluate the robustness of this approach in the whitebox setting by constructing strong optimization-based attacks, and show that the defenses do not provide the claimed security properties. Specifically, we construct a novel attention-based attack algorithm for textual LLMs and apply it to three recent whitebox defenses SecAlign (CCS 2025), SecAlign++, and StruQ (USENIX Security 2025), showing attacks with success rates of up to \textbf{85-95\%} on unseen prompts with modest increase in attacker budget in terms of tokens. Our findings make fundamental progress towards understanding the robustness of prompt injection defenses in the whitebox setting. We release our code and attacks at https://github.com/nishitvp/better_opts_attacks

摘要: 针对大型语言模型（LLM）上的即时注入攻击的一类流行防御依赖于微调以分离指令和数据，以便LLM不会遵循可能存在于数据中的指令。我们通过构建强大的基于优化的攻击来评估这种方法在白盒环境中的稳健性，并表明防御措施不提供声称的安全属性。具体来说，我们为文本LLM构建了一种新颖的基于注意力的攻击算法，并将其应用于最近的三种白盒防御SecAlign（CCS 2025）、SecAlign++和StruQ（USENIX Security 2025），在未见提示上展示了成功率高达\textBF{85-95\%}的攻击，攻击者预算在代币方面略有增加。我们的研究结果在理解白盒环境中即时注射防御的稳健性方面取得了根本性进展。我们在https://github.com/nishitvp/better_opts_attacks上发布我们的代码和攻击



## **46. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

论LLM在对抗性攻击中言语信心的稳健性 cs.CL

Published in NeurIPS 2025

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2507.06489v3) [paper-pdf](https://arxiv.org/pdf/2507.06489v3)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.

摘要: 大型语言模型（LLM）产生的强大言语信心对于LLM的部署至关重要，以帮助确保许多应用程序（包括涉及人机交互的应用程序）的透明度、信任和安全性。在本文中，我们首次对对抗攻击下言语信心的稳健性进行了全面研究。我们通过干扰和基于越狱的方法引入了针对言语信心分数的攻击框架，并证明这些攻击会显着损害言语信心估计并导致答案频繁变化。我们检查了各种提示策略、模型大小和应用领域，揭示了当前的言语自信很脆弱，并且常用的防御技术在很大程度上无效或适得其反。我们的研究结果强调了为LLM中的信心表达设计稳健的机制的必要性，因为即使是微妙的语义保留修改也可能导致反应中的误导性信心。



## **47. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows**

从即时注射到协议漏洞：LLM-Powered AI代理工作流中的威胁 cs.CR

The paper is published in ICT Express (Elsevier)

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2506.23260v2) [paper-pdf](https://arxiv.org/pdf/2506.23260v2)

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Djallel Hamouda, Leandros Maglaras, Abderrahmane Lakas, Merouane Debbah

**Abstract**: Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces enable real-time data retrieval, computation, and multi-step orchestration. However, the rapid growth of plugins, connectors, and inter-agent protocols has outpaced security practices, leading to brittle integrations that rely on ad-hoc authentication, inconsistent schemas, and weak validation. This survey introduces a unified end-to-end threat model for LLM-agent ecosystems, covering host-to-tool and agent-to-agent communications. We systematically categorize more than thirty attack techniques spanning input manipulation, model compromise, system and privacy attacks, and protocol-level vulnerabilities. For each category, we provide a formal threat formulation defining attacker capabilities, objectives, and affected system layers. Representative examples include Prompt-to-SQL injections and the Toxic Agent Flow exploit in GitHub MCP servers. We analyze attack feasibility, review existing defenses, and discuss mitigation strategies such as dynamic trust management, cryptographic provenance tracking, and sandboxed agent interfaces. The framework is validated through expert review and cross-mapping with real-world incidents and public vulnerability repositories, including CVE and NIST NVD. Compared to prior surveys, this work presents the first integrated taxonomy bridging input-level exploits and protocol-layer vulnerabilities in LLM-agent ecosystems, offering actionable guidance for designing secure and resilient agentic AI systems.

摘要: 自主人工智能代理由大型语言模型（LLM）提供支持，具有结构化功能调用接口，可实现实时数据检索、计算和多步骤编排。然而，插件、连接器和代理间协议的快速发展已经超过了安全实践，导致依赖于临时身份验证、不一致的模式和弱验证的脆弱集成。本调查为LLM代理生态系统引入了统一的端到端威胁模型，涵盖主机到工具和代理到代理的通信。我们系统地分类了三十多种攻击技术，涵盖输入操纵、模型妥协、系统和隐私攻击以及协议级漏洞。对于每个类别，我们提供了定义攻击者能力、目标和受影响的系统层的正式威胁公式。代表性示例包括GitHub LCP服务器中的预算到SQL注入和Toxic Agent Flow漏洞利用。我们分析攻击的可行性，审查现有的防御措施，并讨论动态信任管理、加密出处跟踪和沙箱代理接口等缓解策略。该框架通过专家审查和与现实世界事件和公共漏洞存储库（包括UTE和NIH NVD）的交叉映射进行验证。与之前的调查相比，这项工作提出了第一个集成的分类法，弥合了LLM代理生态系统中的输入级漏洞和协议层漏洞，为设计安全且有弹性的代理人工智能系统提供了可操作的指导。



## **48. MoAPT: Mixture of Adversarial Prompt Tuning for Vision-Language Models**

MoAPT：视觉语言模型的对抗性提示调优混合 cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2505.17509v2) [paper-pdf](https://arxiv.org/pdf/2505.17509v2)

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Maoxun Yuan, Jialing Tao, Jiexi Liu, Ranjie Duan, Jie Zhang, Jie Zhang, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) demonstrate excellent generalization capabilities but remain highly susceptible to adversarial examples, posing potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which ultimately results in overfitting. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts yields greater robustness improvements than simply extending the length of a single prompt. Building on this observation, we propose an adversarial tuning method named \textbf{Mixture of Adversarial Prompt Tuning (MoAPT)} to enhance the generalization against various adversarial attacks for VLMs. MoAPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the adversarial images to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific mixture text features aligning with different adversarial image features. Extensive experiments across 11 datasets under different settings show that our method can achieve better adversarial robustness than state-of-the-art approaches.

摘要: 大型预先训练的视觉语言模型（VLM）表现出出色的概括能力，但仍然极易受到对抗性示例的影响，从而构成潜在的安全风险。为了提高VLM对对抗性示例的鲁棒性，提出了对抗性提示调整方法，以在不改变模型参数的情况下将文本特征与对抗性图像特征对齐。然而，当面临各种对抗性攻击时，单个可学习文本提示的概括性不足以与所有对抗性图像特征很好地对齐，这最终会导致过度匹配。为了解决上述挑战，在本文中，我们经验发现，增加学习提示的数量比简单地延长单个提示的长度可以产生更大的鲁棒性改进。在这一观察的基础上，我们提出了一种名为\textBF{混合对抗提示调整（MoAPT）}的对抗性调整方法，以增强针对VLM的各种对抗性攻击的概括性。MoAPT旨在学习混合文本提示以获得更稳健的文本特征。为了进一步增强适应性，我们提出了一种基于对抗图像的条件权重路由器来预测多个学习提示的混合权重，这有助于获得与不同对抗图像特征对齐的样本特定混合文本特征。在不同设置下对11个数据集进行的广泛实验表明，我们的方法可以实现比最先进的方法更好的对抗鲁棒性。



## **49. ExpShield: Safeguarding Web Text from Unauthorized Crawling and LLM Exploitation**

ExpShield：保护Web文本免受未经授权的抓取和LLM利用 cs.CR

18 pages

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2412.21123v3) [paper-pdf](https://arxiv.org/pdf/2412.21123v3)

**Authors**: Ruixuan Liu, Toan Tran, Tianhao Wang, Hongsheng Hu, Shuo Wang, Li Xiong

**Abstract**: As large language models increasingly memorize web-scraped training content, they risk exposing copyrighted or private information. Existing protections require compliance from crawlers or model developers, fundamentally limiting their effectiveness. We propose ExpShield, a proactive self-guard that mitigates memorization while maintaining readability via invisible perturbations, and we formulate it as a constrained optimization problem. Due to the lack of an individual-level risk metric for natural text, we first propose instance exploitation, a metric that measures how much training on a specific text increases the chance of guessing that text from a set of candidates-with zero indicating perfect defense. Directly solving the problem is infeasible for defenders without sufficient knowledge, thus we develop two effective proxy solutions: single-level optimization and synthetic perturbation. To enhance the defense, we reveal and verify the memorization trigger hypothesis, which can help to identify key tokens for memorization. Leveraging this insight, we design targeted perturbations that (i) neutralize inherent trigger tokens to reduce memorization and (ii) introduce artificial trigger tokens to misdirect model memorization. Experiments validate our defense across attacks, model scales, and tasks in language and vision-to-language modeling. Even with privacy backdoor, the Membership Inference Attack (MIA) AUC drops from 0.95 to 0.55 under the defense, and the instance exploitation approaches zero. This suggests that compared to the ideal no-misuse scenario, the risk of exposing a text instance remains nearly unchanged despite its inclusion in the training data.

摘要: 随着大型语言模型越来越多地记住网络抓取的培训内容，它们面临着暴露受版权保护或私人信息的风险。现有的保护要求爬虫或模型开发人员合规，这从根本上限制了其有效性。我们提出了ExpShield，这是一种积极主动的自我保护装置，可以减轻记忆，同时通过不可见的扰动保持可读性，并将其表述为一个受约束的优化问题。由于缺乏自然文本的个人级别风险指标，我们首先提出了实例利用，这是一个衡量对特定文本的训练多少增加了从一组候选文本中猜测该文本的机会的指标-零表示完美防御。对于没有足够知识的防御者来说，直接解决问题是不可行的，因此我们开发了两种有效的代理解决方案：单级优化和合成扰动。为了加强防御，我们揭示并验证了记忆触发假设，这可以帮助识别用于记忆的关键令牌。利用这一见解，我们设计了有针对性的扰动，它们（i）中和固有的触发令牌以减少记忆，以及（ii）引入人工触发令牌以误导模型记忆。实验验证了我们对语言和视觉到语言建模中的攻击、模型规模和任务的防御。即使有隐私后门，在防御下，会员推断攻击（MIA）的UC也从0.95下降到0.55，实例利用率接近零。这表明，与理想的无滥用场景相比，尽管文本实例包含在训练数据中，暴露文本实例的风险几乎保持不变。



## **50. Memory Backdoor Attacks on Neural Networks**

对神经网络的内存后门攻击 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2411.14516v2) [paper-pdf](https://arxiv.org/pdf/2411.14516v2)

**Authors**: Eden Luzon, Guy Amit, Roy Weiss, Torsten Kraub, Alexandra Dmitrienko, Yisroel Mirsky

**Abstract**: Neural networks are often trained on proprietary datasets, making them attractive attack targets. We present a novel dataset extraction method leveraging an innovative training time backdoor attack, allowing a malicious federated learning server to systematically and deterministically extract complete client training samples through a simple indexing process. Unlike prior techniques, our approach guarantees exact data recovery rather than probabilistic reconstructions or hallucinations, provides precise control over which samples are memorized and how many, and shows high capacity and robustness. Infected models output data samples when they receive a patternbased index trigger, enabling systematic extraction of meaningful patches from each clients local data without disrupting global model utility. To address small model output sizes, we extract patches and then recombined them. The attack requires only a minor modification to the training code that can easily evade detection during client-side verification. Hence, this vulnerability represents a realistic FL supply-chain threat, where a malicious server can distribute modified training code to clients and later recover private data from their updates. Evaluations across classifiers, segmentation models, and large language models demonstrate that thousands of sensitive training samples can be recovered from client models with minimal impact on task performance, and a clients entire dataset can be stolen after multiple FL rounds. For instance, a medical segmentation dataset can be extracted with only a 3 percent utility drop. These findings expose a critical privacy vulnerability in FL systems, emphasizing the need for stronger integrity and transparency in distributed training pipelines.

摘要: 神经网络通常在专有数据集上训练，使其成为有吸引力的攻击目标。我们提出了一种新颖的数据集提取方法，利用创新的训练时间后门攻击，允许恶意联邦学习服务器通过简单的索引过程系统性地、确定性地提取完整的客户端训练样本。与现有技术不同，我们的方法保证了准确的数据恢复，而不是概率重建或幻觉，提供了对记忆哪些样本和数量的精确控制，并显示出高容量和鲁棒性。受感染的模型在接收到基于模式的索引触发器时输出数据样本，从而能够从每个客户端的本地数据中系统地提取有意义的补丁，而不会中断全局模型实用性。为了解决小模型输出大小的问题，我们提取补丁，然后重新组合它们。该攻击只需对训练代码进行轻微修改，即可轻松逃避客户端验证期间的检测。因此，该漏洞代表了现实的FL供应链威胁，恶意服务器可以将修改后的训练代码分发给客户端，然后从其更新中恢复私人数据。跨分类器、分段模型和大型语言模型的评估表明，可以从客户端模型中恢复数千个敏感训练样本，对任务性能的影响最小，并且客户端的整个数据集可能会在多轮FL后被盗。例如，医学分割数据集可以仅以3%的效用下降来提取。这些发现暴露了FL系统中的关键隐私漏洞，强调了分布式培训管道需要更强的完整性和透明度。



