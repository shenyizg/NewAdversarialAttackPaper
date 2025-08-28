# Latest Large Language Model Attack Papers
**update at 2025-08-28 10:38:43**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Disabling Self-Correction in Retrieval-Augmented Generation via Stealthy Retriever Poisoning**

通过隐形猎犬中毒禁用检索增强一代的自我纠正 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20083v1) [paper-pdf](http://arxiv.org/pdf/2508.20083v1)

**Authors**: Yanbo Dai, Zhenlan Ji, Zongjie Li, Kuan Li, Shuai Wang

**Abstract**: Retrieval-Augmented Generation (RAG) has become a standard approach for improving the reliability of large language models (LLMs). Prior work demonstrates the vulnerability of RAG systems by misleading them into generating attacker-chosen outputs through poisoning the knowledge base. However, this paper uncovers that such attacks could be mitigated by the strong \textit{self-correction ability (SCA)} of modern LLMs, which can reject false context once properly configured. This SCA poses a significant challenge for attackers aiming to manipulate RAG systems.   In contrast to previous poisoning methods, which primarily target the knowledge base, we introduce \textsc{DisarmRAG}, a new poisoning paradigm that compromises the retriever itself to suppress the SCA and enforce attacker-chosen outputs. This compromisation enables the attacker to straightforwardly embed anti-SCA instructions into the context provided to the generator, thereby bypassing the SCA. To this end, we present a contrastive-learning-based model editing technique that performs localized and stealthy edits, ensuring the retriever returns a malicious instruction only for specific victim queries while preserving benign retrieval behavior. To further strengthen the attack, we design an iterative co-optimization framework that automatically discovers robust instructions capable of bypassing prompt-based defenses. We extensively evaluate DisarmRAG across six LLMs and three QA benchmarks. Our results show near-perfect retrieval of malicious instructions, which successfully suppress SCA and achieve attack success rates exceeding 90\% under diverse defensive prompts. Also, the edited retriever remains stealthy under several detection methods, highlighting the urgent need for retriever-centric defenses.

摘要: 检索增强生成（RAG）已成为提高大型语言模型（LLM）可靠性的标准方法。之前的工作通过毒害知识库来误导RAG系统生成攻击者选择的输出来证明了RAG系统的脆弱性。然而，本文发现，现代LLM的强大\textit{self-correct能力（SCA）}可以减轻此类攻击，一旦配置得当，它就可以拒绝错误的上下文。该SCA对旨在操纵RAG系统的攻击者构成了重大挑战。   与以前的中毒方法，主要针对知识库，我们介绍了\textsc{DisarmRAG}，一个新的中毒范例，妥协检索器本身抑制SCA和执行攻击者选择的输出。这种妥协使攻击者能够直接将反SCA指令嵌入到提供给生成器的上下文中，从而绕过SCA。为此，我们提出了一种基于对比学习的模型编辑技术，该技术执行本地化和隐形编辑，确保检索器仅针对特定的受害者查询返回恶意指令，同时保留良性检索行为。为了进一步加强攻击，我们设计了一个迭代协同优化框架，可以自动发现能够绕过基于密码的防御的强大指令。我们在六个LLM和三个QA基准中广泛评估了DisarmRAG。我们的研究结果表明，近乎完美的恶意指令检索，成功地抑制SCA，并实现攻击成功率超过90%，在不同的防御提示。此外，经过编辑的寻回犬在几种检测方法下仍然保持隐身，突出了以寻回犬为中心的防御的迫切需要。



## **2. Scaling Decentralized Learning with FLock**

利用Flock扩展去中心化学习 cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.15349v2) [paper-pdf](http://arxiv.org/pdf/2507.15349v2)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.

摘要: 由于集中控制的不足以及分散式方案的大量计算和通信负担，大型语言模型（LLM）的微调受到阻碍。虽然典型的标准联邦学习（FL）支持数据隐私，但中央服务器要求会创建单点攻击和中毒攻击的脆弱性。将这一方向的结果推广到异类、无信任环境中的70 B参数模型已被证明是一个巨大但未突破的瓶颈。本文介绍了Flock，这是一个用于安全高效协作LLM微调的去中心化框架。Flock将基于区块链的信任层与经济激励相结合，用安全、可审计的协议取代了中央聚合器，用于不受信任方之间的合作。我们首次对在安全、多域、去中心化的环境中微调70 B LLM进行了实证验证。我们的实验表明，Flock框架可以抵御后门中毒攻击，这些攻击会损害标准FL优化器并促进协同知识转移。由此产生的模型显示对抗性攻击成功率降低了>68%。全局模型还展示了卓越的跨域泛化能力，优于在自己的专业数据上孤立训练的模型。



## **3. Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks**

先发制人：预先合成类似越狱的指令，以增强LLM安全防范潜在攻击 cs.CL

EMNLP 2025 findings

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20038v1) [paper-pdf](http://arxiv.org/pdf/2508.20038v1)

**Authors**: Sheng Liu, Qiang Sheng, Danding Wang, Yang Li, Guang Yang, Juan Cao

**Abstract**: Despite advances in improving large language model(LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.

摘要: 尽管在改进大型语言模型（LLM）以拒绝回答恶意指令方面取得了进展，但广泛使用的LLM仍然容易受到越狱攻击，攻击者生成的指令分布与安全对齐库不同。新的攻击暴露了LLM无法识别不可见的恶意指令，凸显了训练数据和现实世界攻击之间严重的分布不匹配，迫使开发人员进入反应性补丁周期。为了应对这一挑战，我们提出了IMAGINE，这是一个综合框架，利用嵌入空间分布分析来生成类似越狱的指令。这种方法有效地填补了真实越狱模式和安全调整数据库之间的分布差距。IMAGINE遵循迭代优化过程，在迭代中动态演变文本生成分布，从而通过合成数据示例扩大安全对齐数据分布的覆盖范围。基于通过IMAGINE增强的安全对齐的数据库，我们的框架证明了Qwen 2.5、Llama 3.1和Llama 3.2的攻击成功率显着下降，而不会影响其实用性。



## **4. Secure Multi-LLM Agentic AI and Agentification for Edge General Intelligence by Zero-Trust: A Survey**

Zero-Trust为Edge General Intelligence提供的安全多LLM统计人工智能和认证：一项调查 cs.NI

35 pages

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19870v1) [paper-pdf](http://arxiv.org/pdf/2508.19870v1)

**Authors**: Yinqiu Liu, Ruichen Zhang, Haoxiang Luo, Yijing Lin, Geng Sun, Dusit Niyato, Hongyang Du, Zehui Xiong, Yonggang Wen, Abbas Jamalipour, Dong In Kim, Ping Zhang

**Abstract**: Agentification serves as a critical enabler of Edge General Intelligence (EGI), transforming massive edge devices into cognitive agents through integrating Large Language Models (LLMs) and perception, reasoning, and acting modules. These agents collaborate across heterogeneous edge infrastructures, forming multi-LLM agentic AI systems that leverage collective intelligence and specialized capabilities to tackle complex, multi-step tasks. However, the collaborative nature of multi-LLM systems introduces critical security vulnerabilities, including insecure inter-LLM communications, expanded attack surfaces, and cross-domain data leakage that traditional perimeter-based security cannot adequately address. To this end, this survey introduces zero-trust security of multi-LLM in EGI, a paradigmatic shift following the ``never trust, always verify'' principle. We begin by systematically analyzing the security risks in multi-LLM systems within EGI contexts. Subsequently, we present the vision of a zero-trust multi-LLM framework in EGI. We then survey key technical progress to facilitate zero-trust multi-LLM systems in EGI. Particularly, we categorize zero-trust security mechanisms into model- and system-level approaches. The former and latter include strong identification, context-aware access control, etc., and proactive maintenance, blockchain-based management, etc., respectively. Finally, we identify critical research directions. This survey serves as the first systematic treatment of zero-trust applied to multi-LLM systems, providing both theoretical foundations and practical strategies.

摘要: 认证是边缘通用智能（EGI）的关键推动者，通过集成大型语言模型（LLM）以及感知、推理和行为模块，将大型边缘设备转化为认知代理。这些代理跨异类边缘基础设施进行协作，形成多LLM代理人工智能系统，利用集体智慧和专业能力来解决复杂的多步骤任务。然而，多LLM系统的协作性质引入了关键的安全漏洞，包括不安全的LLM间通信、扩展的攻击面以及传统基于边界的安全性无法充分解决的跨域数据泄露。为此，这项调查在EGI中引入了多LLM的零信任安全性，这是遵循“永不信任，始终验证”原则的范式转变。我们首先系统地分析EGI环境下的多LLM系统的安全风险。随后，我们在EGI中提出了零信任多LLM框架的愿景。然后，我们调查了关键技术进展，以促进EGI中的零信任多LLM系统。特别是，我们将零信任安全机制分为模型级和系统级方法。前者和后者包括强识别、上下文感知访问控制等，以及主动维护、基于区块链的管理等，分别最后，我们确定了关键的研究方向。这项调查是首次系统地处理应用于多LLM系统的零信任，提供了理论基础和实践策略。



## **5. EnvInjection: Environmental Prompt Injection Attack to Multi-modal Web Agents**

EnvInjecting：对多模式Web代理的环境提示注入攻击 cs.LG

EMNLP 2025 main

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2505.11717v2) [paper-pdf](http://arxiv.org/pdf/2505.11717v2)

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. Environmental prompt injection attacks manipulate the environment to induce the web agent to perform a specific, attacker-chosen action--denoted as the target action. However, existing attacks suffer from limited effectiveness or stealthiness, or are impractical in real-world settings. In this work, we propose EnvInjection, a new attack that addresses these limitations. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the target action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple webpage datasets shows that EnvInjection is highly effective and significantly outperforms existing baselines.

摘要: 基于多模式大型语言模型（MLLM）的Web代理通过基于网页屏幕截图生成动作来与网页环境交互。环境提示注入攻击操纵环境以诱导Web代理执行特定的攻击者选择的操作--称为目标操作。然而，现有的攻击的有效性或隐蔽性有限，或者在现实世界环境中不切实际。在这项工作中，我们提出了EnvInjectation，这是一种解决这些限制的新攻击。我们的攻击对渲染网页的原始像素值添加了扰动。这些受干扰的像素被映射到屏幕截图后，受干扰会导致Web代理执行目标操作。我们将寻找扰动的任务定义为优化问题。解决这个问题的一个关键挑战是原始像素值和屏幕截图之间的映射是不可微的，因此很难将梯度反向传播到扰动。为了克服这个问题，我们训练神经网络来逼近映射，并应用投影梯度下降来解决重新制定的优化问题。对多个网页数据集的广泛评估表明，EnvInjecting非常有效，并且显着优于现有基线。



## **6. Safety Alignment Should Be Made More Than Just A Few Attention Heads**

安全调整不应仅仅是一些注意力 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19697v1) [paper-pdf](http://arxiv.org/pdf/2508.19697v1)

**Authors**: Chao Huang, Zefeng Zhang, Juewei Yue, Quangang Li, Chuang Zhang, Tingwen Liu

**Abstract**: Current safety alignment for large language models(LLMs) continues to present vulnerabilities, given that adversarial prompting can effectively bypass their safety measures.Our investigation shows that these safety mechanisms predominantly depend on a limited subset of attention heads: removing or ablating these heads can severely compromise model safety. To identify and evaluate these safety-critical components, we introduce RDSHA, a targeted ablation method that leverages the model's refusal direction to pinpoint attention heads mostly responsible for safety behaviors. Further analysis shows that existing jailbreak attacks exploit this concentration by selectively bypassing or manipulating these critical attention heads. To address this issue, we propose AHD, a novel training strategy designed to promote the distributed encoding of safety-related behaviors across numerous attention heads. Experimental results demonstrate that AHD successfully distributes safety-related capabilities across more attention heads. Moreover, evaluations under several mainstream jailbreak attacks show that models trained with AHD exhibit considerably stronger safety robustness, while maintaining overall functional utility.

摘要: 目前的大型语言模型（LLM）的安全对齐仍然存在漏洞，因为对抗性提示可以有效地绕过它们的安全措施。我们的调查表明，这些安全机制主要依赖于有限的注意头子集：删除或消融这些头会严重危及模型安全。为了识别和评估这些安全关键组件，我们引入了RDSHA，这是一种有针对性的消融方法，它利用模型的拒绝方向来确定主要负责安全行为的注意力。进一步的分析表明，现有的越狱攻击通过选择性地绕过或操纵这些关键注意力头部来利用这种集中。为了解决这个问题，我们提出了AHD，一种新的训练策略，旨在促进分布式编码的安全相关的行为在众多的注意头。实验结果表明，AHD成功地将安全相关功能分配给更多的注意力头。此外，在几种主流越狱攻击下的评估表明，用AHD训练的模型表现出更强的安全鲁棒性，同时保持整体功能效用。



## **7. PromptKeeper: Safeguarding System Prompts for LLMs**

PretKeeper：保护LLM的系统预算 cs.CR

Accepted to the Findings of EMNLP 2025. 17 pages, 6 figures, 3 tables

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2412.13426v3) [paper-pdf](http://arxiv.org/pdf/2412.13426v3)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: System prompts are widely used to guide the outputs of large language models (LLMs). These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a defense mechanism designed to safeguard system prompts by tackling two core challenges: reliably detecting leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon leakage detected, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.

摘要: 系统提示被广泛用于指导大型语言模型（LLM）的输出。这些提示通常包含业务逻辑和敏感信息，因此对其的保护至关重要。然而，对抗性甚至常规用户查询都可能利用LLM漏洞来暴露这些隐藏的提示。为了解决这个问题，我们提出了Inbox Keeper，这是一种防御机制，旨在通过解决两个核心挑战来保护系统提示：可靠地检测泄漏和减轻发生泄漏时的侧通道漏洞。通过将检测视为假设测试问题，SpectKeeper有效地识别显式和微妙的泄漏。检测到泄漏后，它会使用虚拟提示重新生成响应，确保在不存在泄漏时输出与典型交互没有区别。EntKeeper确保针对通过对抗性或常规查询的即时提取攻击提供强大的保护，同时在良性用户交互期间保留对话能力和运行时效率。



## **8. MEraser: An Effective Fingerprint Erasure Approach for Large Language Models**

MEraser：大型语言模型的有效指纹擦除方法 cs.CR

Accepted by ACL 2025, Main Conference, Long Paper

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2506.12551v2) [paper-pdf](http://arxiv.org/pdf/2506.12551v2)

**Authors**: Jingxuan Zhang, Zhenhua Xu, Rui Hu, Wenpeng Xing, Xuhong Zhang, Meng Han

**Abstract**: Large Language Models (LLMs) have become increasingly prevalent across various sectors, raising critical concerns about model ownership and intellectual property protection. Although backdoor-based fingerprinting has emerged as a promising solution for model authentication, effective attacks for removing these fingerprints remain largely unexplored. Therefore, we present Mismatched Eraser (MEraser), a novel method for effectively removing backdoor-based fingerprints from LLMs while maintaining model performance. Our approach leverages a two-phase fine-tuning strategy utilizing carefully constructed mismatched and clean datasets. Through extensive evaluation across multiple LLM architectures and fingerprinting methods, we demonstrate that MEraser achieves complete fingerprinting removal while maintaining model performance with minimal training data of fewer than 1,000 samples. Furthermore, we introduce a transferable erasure mechanism that enables effective fingerprinting removal across different models without repeated training. In conclusion, our approach provides a practical solution for fingerprinting removal in LLMs, reveals critical vulnerabilities in current fingerprinting techniques, and establishes comprehensive evaluation benchmarks for developing more resilient model protection methods in the future.

摘要: 大型语言模型（LLM）在各个领域变得越来越普遍，引发了人们对模型所有权和知识产权保护的严重担忧。尽管基于后门的指纹识别已成为模型认证的一种有希望的解决方案，但用于删除这些指纹的有效攻击在很大程度上仍然没有被探索。因此，我们提出了Mmatched Eraser（MEraser），这是一种新型方法，可以有效地从LLM中删除基于后门的指纹，同时保持模型性能。我们的方法利用两阶段微调策略，利用精心构建的不匹配且干净的数据集。通过对多种LLM架构和指纹识别方法的广泛评估，我们证明MEraser可以通过少于1，000个样本的最少训练数据实现完全指纹识别，同时保持模型性能。此外，我们引入了一种可转移的擦除机制，可以在不同模型之间有效地去除指纹，而无需重复训练。总之，我们的方法为LLM中的指纹删除提供了一种实用的解决方案，揭示了当前指纹技术中的关键漏洞，并为未来开发更具弹性的模型保护方法建立了全面的评估基准。



## **9. An Investigation on Group Query Hallucination Attacks**

群查询幻觉攻击调查 cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19321v1) [paper-pdf](http://arxiv.org/pdf/2508.19321v1)

**Authors**: Kehao Miao, Xiaolong Jin

**Abstract**: With the widespread use of large language models (LLMs), understanding their potential failure modes during user interactions is essential. In practice, users often pose multiple questions in a single conversation with LLMs. Therefore, in this study, we propose Group Query Attack, a technique that simulates this scenario by presenting groups of queries to LLMs simultaneously. We investigate how the accumulated context from consecutive prompts influences the outputs of LLMs. Specifically, we observe that Group Query Attack significantly degrades the performance of models fine-tuned on specific tasks. Moreover, we demonstrate that Group Query Attack induces a risk of triggering potential backdoors of LLMs. Besides, Group Query Attack is also effective in tasks involving reasoning, such as mathematical reasoning and code generation for pre-trained and aligned models.

摘要: 随着大型语言模型（LLM）的广泛使用，了解其在用户交互期间的潜在故障模式至关重要。在实践中，用户经常在与LLM的一次对话中提出多个问题。因此，在这项研究中，我们提出了组查询攻击，这是一种通过同时向LLM呈现组查询来模拟这种场景的技术。我们研究连续提示积累的上下文如何影响LLM的输出。具体来说，我们观察到组查询攻击会显着降低针对特定任务进行微调的模型的性能。此外，我们证明了组查询攻击会引发触发LLM潜在后门的风险。此外，群查询攻击在涉及推理的任务中也很有效，例如数学推理和预训练和对齐模型的代码生成。



## **10. RePPL: Recalibrating Perplexity by Uncertainty in Semantic Propagation and Language Generation for Explainable QA Hallucination Detection**

RePPL：通过语义传播和语言生成中的不确定性重新校准困惑，以实现可解释QA幻觉检测 cs.CL

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2505.15386v2) [paper-pdf](http://arxiv.org/pdf/2505.15386v2)

**Authors**: Yiming Huang, Junyan Zhang, Zihao Wang, Biquan Bie, Yunzhong Qiu, Yi R. Fung, Xinlei He

**Abstract**: Large Language Models (LLMs) have become powerful, but hallucinations remain a vital obstacle to their trustworthy use. While previous works improved the capability of hallucination detection by measuring uncertainty, they all lack the ability to explain the provenance behind why hallucinations occur, i.e., which part of the inputs tends to trigger hallucinations. Recent works on the prompt attack indicate that uncertainty exists in semantic propagation, where attention mechanisms gradually fuse local token information into high-level semantics across layers. Meanwhile, uncertainty also emerges in language generation, due to its probability-based selection of high-level semantics for sampled generations. Based on that, we propose RePPL to recalibrate uncertainty measurement by these two aspects, which dispatches explainable uncertainty scores to each token and aggregates in Perplexity-style Log-Average form as total score. Experiments show that our method achieves the best comprehensive detection performance across various QA datasets on advanced models (average AUC of 0.833), and our method is capable of producing token-level uncertainty scores as explanations for the hallucination. Leveraging these scores, we preliminarily find the chaotic pattern of hallucination and showcase its promising usage.

摘要: 大型语言模型（LLM）已经变得强大，但幻觉仍然是其值得信赖使用的重要障碍。虽然之前的作品通过测量不确定性来提高了幻觉检测的能力，但它们都缺乏解释幻觉发生背后来源的能力，即这部分输入往往会引发幻觉。最近关于提示攻击的研究表明，语义传播中存在不确定性，其中注意力机制逐渐将本地令牌信息融合到跨层的高级语义中。与此同时，由于语言生成对采样世代的高级语义基于概率选择，因此也出现了不确定性。在此基础上，我们提出RePPL通过这两个方面重新校准不确定性测量，将可解释的不确定性分数分配到每个代币，并以困惑式的Log-Average形式汇总为总分。实验表明，我们的方法在高级模型上的各种QA数据集中实现了最佳的综合检测性能（平均曲线下面积为0.833），并且我们的方法能够产生符号级的不确定性分数作为幻觉的解释。利用这些分数，我们初步发现了幻觉的混乱模式，并展示了其有希望的用途。



## **11. The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization**

基于LLM的数据重建的双刃剑：理解和缓解词级差异隐私文本清理中的上下文漏洞 cs.CR

15 pages, 4 figures, 8 tables. Accepted to WPES @ CCS 2025

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18976v1) [paper-pdf](http://arxiv.org/pdf/2508.18976v1)

**Authors**: Stephen Meisenbacher, Alexandra Klymenko, Andreea-Elena Bodea, Florian Matthes

**Abstract**: Differentially private text sanitization refers to the process of privatizing texts under the framework of Differential Privacy (DP), providing provable privacy guarantees while also empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP text sanitization methods operating at the word level exhibit a number of shortcomings, among them the tendency to leave contextual clues from the original texts due to randomization during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual vulnerability}$. Given the powerful contextual understanding and inference capabilities of Large Language Models (LLMs), we explore to what extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We expand on previous work not only in the use of advanced LLMs, but also in testing a broader range of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed infer original semantics and sometimes degrade empirical privacy protections, they can also be used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we propose recommendations for using LLM data reconstruction as a post-processing step, serving to increase privacy protection by thinking adversarially.

摘要: 差异隐私文本清理是指在差异隐私（DP）框架下将文本私有化的过程，提供可证明的隐私保证，同时还根据经验防御试图损害隐私的对手。尽管它们很简单，但在词级操作的DP文本清理方法表现出许多缺点，其中包括由于清理期间的随机性，倾向于从原始文本中留下上下文线索$\unicode{x2013}$我们将其称为$\textit{contextual vulnerability}$。鉴于大型语言模型（LLM）强大的上下文理解和推理能力，我们探索可以在多大程度上利用LLM来利用DP清理文本的上下文脆弱性。我们不仅扩展了之前的工作，还扩展了高级LLM的使用，还扩展了各种隐私级别的更广泛的清理机制。我们的实验揭示了基于LLM的数据重建攻击对隐私和实用性的双刃剑效应：虽然LLM确实可以推断原始语义，有时会降低经验隐私保护，但它们也可以被永久使用，以提高DP净化文本的质量和隐私。根据我们的研究结果，我们提出了使用LLM数据重建作为后处理步骤的建议，通过敌对思维来增加隐私保护。



## **12. SDGO: Self-Discrimination-Guided Optimization for Consistent Safety in Large Language Models**

SDGO：自我辨别引导的大型语言模型中一致安全性优化 cs.CL

Accepted by EMNLP 2025 (Main Conference), 15 pages, 4 figures, 6  tables

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.15648v2) [paper-pdf](http://arxiv.org/pdf/2508.15648v2)

**Authors**: Peng Ding, Wen Sun, Dailin Li, Wei Zou, Jiaming Wang, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs) excel at various natural language processing tasks but remain vulnerable to jailbreaking attacks that induce harmful content generation. In this paper, we reveal a critical safety inconsistency: LLMs can more effectively identify harmful requests as discriminators than defend against them as generators. This insight inspires us to explore aligning the model's inherent discrimination and generation capabilities. To this end, we propose SDGO (Self-Discrimination-Guided Optimization), a reinforcement learning framework that leverages the model's own discrimination capabilities as a reward signal to enhance generation safety through iterative self-improvement. Our method does not require any additional annotated data or external models during the training phase. Extensive experiments demonstrate that SDGO significantly improves model safety compared to both prompt-based and training-based baselines while maintaining helpfulness on general benchmarks. By aligning LLMs' discrimination and generation capabilities, SDGO brings robust performance against out-of-distribution (OOD) jailbreaking attacks. This alignment achieves tighter coupling between these two capabilities, enabling the model's generation capability to be further enhanced with only a small amount of discriminative samples. Our code and datasets are available at https://github.com/NJUNLP/SDGO.

摘要: 大型语言模型（LLM）擅长各种自然语言处理任务，但仍然容易受到导致有害内容生成的越狱攻击。在本文中，我们揭示了一个关键的安全不一致性：LLM可以更有效地识别有害请求作为识别器，而不是作为生成器来防御有害请求。这一见解激励我们探索如何调整模型的固有歧视和生成能力。为此，我们提出了SDGO（自我歧视引导优化），这是一种强化学习框架，它利用模型自身的歧视能力作为奖励信号，通过迭代自我改进来增强发电安全性。我们的方法在训练阶段不需要任何额外的注释数据或外部模型。大量实验表明，与基于预算和基于培训的基线相比，SDGO显着提高了模型安全性，同时保持了一般基准的帮助性。通过协调LLM的区分和生成能力，SDGO为抵御分发外（OOD）越狱攻击带来了强劲的性能。这种对齐实现了这两种功能之间的更紧密耦合，使模型的生成能力能够进一步增强，只需少量的判别样本。我们的代码和数据集可以在https://github.com/NJUNLP/SDGO上找到。



## **13. sudoLLM: On Multi-role Alignment of Language Models**

sudoLLM：关于语言模型的多角色对齐 cs.CL

Accepted to EMNLP 2025 (findings)

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2505.14607v3) [paper-pdf](http://arxiv.org/pdf/2505.14607v3)

**Authors**: Soumadeep Saha, Akshay Chaturvedi, Joy Mahapatra, Utpal Garain

**Abstract**: User authorization-based access privileges are a key feature in many safety-critical systems, but have not been extensively studied in the large language model (LLM) realm. In this work, drawing inspiration from such access control systems, we introduce sudoLLM, a novel framework that results in multi-role aligned LLMs, i.e., LLMs that account for, and behave in accordance with, user access rights. sudoLLM injects subtle user-based biases into queries and trains an LLM to utilize this bias signal in order to produce sensitive information if and only if the user is authorized. We present empirical results demonstrating that this approach shows substantially improved alignment, generalization, resistance to prefix-based jailbreaking attacks, and ``fails-closed''. The persistent tension between the language modeling objective and safety alignment, which is often exploited to jailbreak LLMs, is somewhat resolved with the aid of the injected bias signal. Our framework is meant as an additional security layer, and complements existing guardrail mechanisms for enhanced end-to-end safety with LLMs.

摘要: 基于用户授权的访问特权是许多安全关键系统的一个关键功能，但在大型语言模型（LLM）领域尚未进行广泛研究。在这项工作中，我们从此类访问控制系统中汲取灵感，引入了sudoLLM，这是一种新颖的框架，可以产生多角色对齐的LLM，即负责用户访问权限并按照用户访问权限行事的LLM。sudoLLM将微妙的基于用户的偏见注入到查询中，并训练LLM利用此偏见信号，以便在且仅在用户获得授权的情况下生成敏感信息。我们提出的经验结果表明，这种方法显示出大幅改善的对齐性、概括性、对基于后缀的越狱攻击的抵抗力和“失败关闭”。语言建模目标和安全对齐之间的持续紧张关系（通常被用来越狱LLM）在注入的偏见信号的帮助下在一定程度上得到了解决。我们的框架旨在作为额外的安全层，并补充现有的护栏机制，通过LLM增强端到端安全性。



## **14. FALCON: Autonomous Cyber Threat Intelligence Mining with LLMs for IDS Rule Generation**

RISKCON：使用LLM进行IDS规则生成的自主网络威胁情报挖掘 cs.CR

11 pages, 5 figures, 4 tables

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18684v1) [paper-pdf](http://arxiv.org/pdf/2508.18684v1)

**Authors**: Shaswata Mitra, Azim Bazarov, Martin Duclos, Sudip Mittal, Aritran Piplai, Md Rayhanur Rahman, Edward Zieglar, Shahram Rahimi

**Abstract**: Signature-based Intrusion Detection Systems (IDS) detect malicious activities by matching network or host activity against predefined rules. These rules are derived from extensive Cyber Threat Intelligence (CTI), which includes attack signatures and behavioral patterns obtained through automated tools and manual threat analysis, such as sandboxing. The CTI is then transformed into actionable rules for the IDS engine, enabling real-time detection and prevention. However, the constant evolution of cyber threats necessitates frequent rule updates, which delay deployment time and weaken overall security readiness. Recent advancements in agentic systems powered by Large Language Models (LLMs) offer the potential for autonomous IDS rule generation with internal evaluation. We introduce FALCON, an autonomous agentic framework that generates deployable IDS rules from CTI data in real-time and evaluates them using built-in multi-phased validators. To demonstrate versatility, we target both network (Snort) and host-based (YARA) mediums and construct a comprehensive dataset of IDS rules with their corresponding CTIs. Our evaluations indicate FALCON excels in automatic rule generation, with an average of 95% accuracy validated by qualitative evaluation with 84% inter-rater agreement among multiple cybersecurity analysts across all metrics. These results underscore the feasibility and effectiveness of LLM-driven data mining for real-time cyber threat mitigation.

摘要: 基于签名的入侵检测系统（IDS）通过将网络或主机活动与预定义规则匹配来检测恶意活动。这些规则源自广泛的网络威胁情报（RTI），其中包括通过自动化工具和手动威胁分析（例如沙箱）获得的攻击签名和行为模式。然后，RTI转换为IDS引擎的可操作规则，从而实现实时检测和预防。然而，网络威胁的不断演变需要频繁的规则更新，这会推迟部署时间并削弱整体安全准备状态。由大型语言模型（LLM）支持的代理系统的最新进展为通过内部评估的自主IDS规则生成提供了潜力。我们引入了CLARCON，这是一个自主代理框架，可以根据RTI数据实时生成可部署的IDS规则，并使用内置的多阶段验证器对其进行评估。为了展示多功能性，我们针对网络（Snort）和基于主机（YARA）的媒体，并利用相应的RTI构建IDS规则的全面数据集。我们的评估表明，CLARCON在自动规则生成方面表现出色，通过定性评估验证了平均95%的准确性，多名网络安全分析师在所有指标上的一致性为84%。这些结果强调了LLM驱动的数据挖掘用于实时网络威胁缓解的可行性和有效性。



## **15. Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework**

LLM可以处理WebShell检测吗？使用行为功能感知框架克服检测挑战 cs.CR

Published as a conference paper at COLM 2025

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2504.13811v3) [paper-pdf](http://arxiv.org/pdf/2504.13811v3)

**Authors**: Feijiang Han, Jiaming Zhang, Chuyi Deng, Jianheng Tang, Yunhuai Liu

**Abstract**: WebShell attacks, where malicious scripts are injected into web servers, pose a significant cybersecurity threat. Traditional ML and DL methods are often hampered by challenges such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models have emerged as powerful alternatives for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that, stemming from their distinct analytical strategies, larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all baseline models lag behind previous SOTA methods. With the application of BFAD, the performance of all LLMs improves significantly, yielding an average F1 score increase of 13.82%. Notably, larger models now outperform SOTA benchmarks, while smaller models such as Qwen-2.5-Coder-3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection and provides solutions to address the challenges in this task.

摘要: WebShell攻击（恶意脚本被注入网络服务器）构成了重大的网络安全威胁。传统的ML和DL方法经常受到需要大量训练数据、灾难性遗忘和较差的概括性等挑战的阻碍。最近，大型语言模型已成为代码相关任务的强大替代方案，但它们在WebShell检测中的潜力仍然没有得到充分的开发。在本文中，我们做出了两项贡献：（1）对七种LLM进行了全面评估，包括GPT-4、LLaMA 3.1 70 B和Qwen 2.5变体，使用26.59 K个PHP脚本的数据集针对传统的基于序列和图形的方法进行基准测试，以及（2）行为功能感知检测（BFAD）框架，旨在解决将LLM应用于该领域的特定挑战。我们的框架集成了三个组件：隔离恶意PHP函数调用的关键函数过滤器、捕获最具行为指示性的代码段的上下文感知代码提取策略，以及通过优先考虑最相关的演示来增强上下文学习的加权行为函数剖析基于区分性功能级配置文件。我们的结果表明，由于其不同的分析策略，较大的LLM可以实现近乎完美的精确度，但召回率较低，而较小的模型则表现出相反的权衡。然而，所有基线模型都落后于之前的SOTA方法。随着BFAD的应用，所有LLM的性能都显着提高，F1平均得分提高了13.82%。值得注意的是，大型型号的性能现在优于SOTA基准，而Qwen-2.5-Coder-3B等小型型号的性能与传统方法相比具有竞争力。这项工作是第一个探索LLM用于WebShell检测的可行性和局限性的工作，并提供了解决方案来应对这项任务中的挑战。



## **16. Membership Inference Attacks on LLM-based Recommender Systems**

对基于LLM的推荐系统的成员推断攻击 cs.IR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18665v1) [paper-pdf](http://arxiv.org/pdf/2508.18665v1)

**Authors**: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen

**Abstract**: Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.

摘要: 基于大型语言模型（LLM）的推荐系统（RecSys）可以灵活地调整推荐系统以适应不同的领域。它利用上下文学习（ICL），即提示来定制推荐功能，其中包括敏感的历史用户特定项目交互，例如，隐性反馈，例如点击的项目或明确的产品评论。此类私人信息可能会受到新型隐私攻击。然而，对于这个重要问题还没有进行任何研究。我们设计了四种成员资格推断攻击（MIA），旨在揭示受害者的历史互动是否被系统提示使用。它们是\r {直接询问、幻觉、相似性和中毒攻击}，每一种都利用了LLM或RecSys的独特功能。我们在三个用于开发ICL-LLM RecSys的LLM和两个著名的RecSys基准数据集上仔细评估了它们。结果证实，LLM RecSys上的MIA威胁是现实的：直接询问和中毒攻击显示出显着较高的攻击优势。我们还分析了影响这些攻击的因素，例如系统提示中的射击次数以及受害者在射击中的位置。



## **17. Large Language Model-Based Framework for Explainable Cyberattack Detection in Automatic Generation Control Systems**

自动发电控制系统中基于大语言模型的可解释网络攻击检测框架 cs.CR

Accepted Paper

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2507.22239v2) [paper-pdf](http://arxiv.org/pdf/2507.22239v2)

**Authors**: Muhammad Sharshar, Ahmad Mohammad Saber, Davor Svetinovic, Amr M. Youssef, Deepa Kundur, Ehab F. El-Saadany

**Abstract**: The increasing digitization of smart grids has improved operational efficiency but also introduced new cybersecurity vulnerabilities, such as False Data Injection Attacks (FDIAs) targeting Automatic Generation Control (AGC) systems. While machine learning (ML) and deep learning (DL) models have shown promise in detecting such attacks, their opaque decision-making limits operator trust and real-world applicability. This paper proposes a hybrid framework that integrates lightweight ML-based attack detection with natural language explanations generated by Large Language Models (LLMs). Classifiers such as LightGBM achieve up to 95.13% attack detection accuracy with only 0.004 s inference latency. Upon detecting a cyberattack, the system invokes LLMs, including GPT-3.5 Turbo, GPT-4 Turbo, and GPT-4o mini, to generate human-readable explanation of the event. Evaluated on 100 test samples, GPT-4o mini with 20-shot prompting achieved 93% accuracy in identifying the attack target, a mean absolute error of 0.075 pu in estimating attack magnitude, and 2.19 seconds mean absolute error (MAE) in estimating attack onset. These results demonstrate that the proposed framework effectively balances real-time detection with interpretable, high-fidelity explanations, addressing a critical need for actionable AI in smart grid cybersecurity.

摘要: 智能电网日益数字化提高了运营效率，但也带来了新的网络安全漏洞，例如针对自动发电控制（AGC）系统的虚假数据注入攻击（FDIA）。虽然机器学习（ML）和深度学习（DL）模型在检测此类攻击方面表现出了希望，但其不透明的决策限制了操作员的信任和现实世界的适用性。本文提出了一种混合框架，集成了轻量级的ML为基础的攻击检测与自然语言解释产生的大语言模型（LLM）。LightGBM等分类器的攻击检测准确率高达95.13%，推理延迟仅为0.004 s。检测到网络攻击后，系统会调用LLM（包括GPT-3.5涡轮、GPT-4涡轮和GPT-4 o mini）来生成人类可读的事件解释。经过100个测试样本的评估，带20发提示的GPT-4 o mini识别攻击目标的准确率达到了93%，估计攻击幅度的平均绝对误差为0.075 pu，估计攻击开始的平均绝对误差为2.19秒。这些结果表明，拟议的框架有效地平衡了实时检测与可解释的高保真解释，满足了智能电网网络安全中对可操作人工智能的迫切需求。



## **18. Prefill-level Jailbreak: A Black-Box Risk Analysis of Large Language Models**

预填充级越狱：大型语言模型的黑匣子风险分析 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2504.21038v2) [paper-pdf](http://arxiv.org/pdf/2504.21038v2)

**Authors**: Yakai Li, Jiekang Hu, Weiduan Sang, Luping Ma, Dongsheng Nie, Weijuan Zhang, Aimin Yu, Yi Su, Qingjia Huang, Qihang Zhou

**Abstract**: Large Language Models face security threats from jailbreak attacks. Existing research has predominantly focused on prompt-level attacks while largely ignoring the underexplored attack surface of user-controlled response prefilling. This functionality allows an attacker to dictate the beginning of a model's output, thereby shifting the attack paradigm from persuasion to direct state manipulation.In this paper, we present a systematic black-box security analysis of prefill-level jailbreak attacks. We categorize these new attacks and evaluate their effectiveness across fourteen language models. Our experiments show that prefill-level attacks achieve high success rates, with adaptive methods exceeding 99% on several models. Token-level probability analysis reveals that these attacks work through initial-state manipulation by changing the first-token probability from refusal to compliance.Furthermore, we show that prefill-level jailbreak can act as effective enhancers, increasing the success of existing prompt-level attacks by 10 to 15 percentage points. Our evaluation of several defense strategies indicates that conventional content filters offer limited protection. We find that a detection method focusing on the manipulative relationship between the prompt and the prefill is more effective. Our findings reveal a gap in current LLM safety alignment and highlight the need to address the prefill attack surface in future safety training.

摘要: 大型语言模型面临越狱攻击的安全威胁。现有的研究主要集中在预算级攻击上，而很大程度上忽视了用户控制响应预填充的未充分探索的攻击表面。该功能允许攻击者决定模型输出的开始，从而将攻击范式从说服转变为直接状态操纵。在本文中，我们对预填充级越狱攻击进行了系统性的黑匣子安全分析。我们对这些新攻击进行了分类，并评估了它们在十四种语言模型中的有效性。我们的实验表明，预填充级攻击的成功率很高，自适应方法在几种模型上的成功率超过了99%。代币级概率分析表明，这些攻击通过初始状态操纵进行，将第一代币概率从拒绝改为遵守。此外，我们表明预填充级越狱可以充当有效的增强剂，将现有预算级攻击的成功率提高10到15个百分点。我们对几种防御策略的评估表明，传统内容过滤器提供的保护有限。我们发现，专注于提示和预填充之间操纵关系的检测方法更有效。我们的研究结果揭示了当前LLM安全调整方面的差距，并强调了在未来的安全培训中解决预填充攻击面的必要性。



## **19. Defending Against Prompt Injection With a Few DefensiveTokens**

使用一些防御代币来防御即时注射 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.07974v2) [paper-pdf](http://arxiv.org/pdf/2507.07974v2)

**Authors**: Sizhe Chen, Yizhu Wang, Nicholas Carlini, Chawin Sitawarin, David Wagner

**Abstract**: When large language model (LLM) systems interact with external data to perform complex tasks, a new attack, namely prompt injection, becomes a significant threat. By injecting instructions into the data accessed by the system, the attacker is able to override the initial user task with an arbitrary task directed by the attacker. To secure the system, test-time defenses, e.g., defensive prompting, have been proposed for system developers to attain security only when needed in a flexible manner. However, they are much less effective than training-time defenses that change the model parameters. Motivated by this, we propose DefensiveToken, a test-time defense with prompt injection robustness comparable to training-time alternatives. DefensiveTokens are newly inserted as special tokens, whose embeddings are optimized for security. In security-sensitive cases, system developers can append a few DefensiveTokens before the LLM input to achieve security with a minimal utility drop. In scenarios where security is less of a concern, developers can simply skip DefensiveTokens; the LLM system remains the same as there is no defense, generating high-quality responses. Thus, DefensiveTokens, if released alongside the model, allow a flexible switch between the state-of-the-art (SOTA) utility and almost-SOTA security at test time. The code is available at https://github.com/Sizhe-Chen/DefensiveToken.

摘要: 当大型语言模型（LLM）系统与外部数据交互以执行复杂任务时，一种新的攻击（即提示注入）将成为重大威胁。通过将指令注入系统访问的数据中，攻击者能够用攻击者指示的任意任务覆盖初始用户任务。为了保护系统，测试时防御措施，例如防御性提示已被建议供系统开发人员仅在需要时以灵活的方式获得安全性。然而，它们比改变模型参数的训练时防御有效得多。出于此动机，我们提出了DefensiveToken，这是一种测试时防御，具有与训练时替代方案相当的即时注入鲁棒性。DefensiveTokens作为特殊令牌新插入，其嵌入针对安全性进行了优化。在安全敏感的情况下，系统开发人员可以在LLM输入之前添加一些DefensiveTokens，以最小的实用程序下降来实现安全性。在安全性不太值得关注的场景中，开发人员可以简单地跳过DefensiveTokens; LLM系统由于没有防御而保持不变，从而生成高质量的响应。因此，DefensiveTokens如果与该模型一起发布，将允许在测试时在最先进的（SOTA）实用程序和几乎SOTA安全性之间灵活切换。该代码可在https://github.com/Sizhe-Chen/DefensiveToken上获取。



## **20. Confidential Prompting: Privacy-preserving LLM Inference on Cloud**

机密认证：云上的隐私保护LLM推理 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2409.19134v4) [paper-pdf](http://arxiv.org/pdf/2409.19134v4)

**Authors**: Caihua Li, In Gim, Lin Zhong

**Abstract**: This paper introduces a vision of confidential prompting: securing user prompts from untrusted, cloud-hosted large language model (LLM) provider while preserving model confidentiality, output invariance, and compute efficiency. As a first step toward this vision, we present Obfuscated Secure Partitioned Decoding (OSPD), a system built on two key innovations. First, Secure Partitioned Decoding (SPD) isolates user prompts within per-user processes residing in a confidential virtual machine (CVM) on the cloud, which are inaccessible for the cloud LLM while allowing it to generate tokens efficiently. Second, Prompt Obfuscation (PO) introduces a novel cryptographic technique that enhances SPD resilience against advanced prompt reconstruction attacks. Together, these innovations ensure OSPD protects both prompt and model confidentiality while maintaining service functionality. OSPD enables practical, privacy-preserving cloud-hosted LLM inference for sensitive applications, such as processing personal data, clinical records, and financial documents.

摘要: 本文介绍了保密提示的愿景：保护来自不受信任的云托管大型语言模型（LLM）提供商的用户提示，同时保留模型机密性、输出不变性和计算效率。作为实现这一愿景的第一步，我们提出了模糊安全分区解码（OSPD），这是一个基于两项关键创新的系统。首先，安全分区解码（SPD）将用户提示隔离在云上机密虚拟机（CGM）中驻留的每个用户进程中，云LLM无法访问这些进程，同时允许其高效地生成令牌。其次，即时混淆（PO）引入了一种新型加密技术，可以增强SPD抵御高级即时重建攻击的弹性。这些创新共同确保OSPD在维护服务功能的同时保护即时和模型机密性。OSPD为敏感应用程序（例如处理个人数据、临床记录和财务文档）提供实用的、保护隐私的云托管LLM推断。



## **21. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

引导对话动力学，增强抵御多回合越狱攻击的稳健性 cs.CL

23 pages, 10 figures, 11 tables

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2503.00187v2) [paper-pdf](http://arxiv.org/pdf/2503.00187v2)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are shown to be vulnerable to jailbreaking attacks where adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment, prompt-based steering and lightweight LLM guardrails baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off among safety, helpfulness and over-refusal. Check out the website here https://sites.google.com/view/llm-nbf/home . Our code is available on https://github.com/HanjiangHu/NBF-LLM .

摘要: 事实证明，大型语言模型（LLM）很容易受到越狱攻击，其中对抗性提示旨在引发有害反应。虽然现有的防御措施通过检测和过滤不安全的输入有效地减轻了单回合攻击，但它们无法对抗利用多次交互中的上下文漂移的多回合越狱，从而逐渐导致LLM远离安全行为。为了应对这一挑战，我们提出了一个基于安全控制理论的安全引导框架，确保多回合对话中不变的安全性。我们的方法使用状态空间表示对与LLM的对话进行建模，并引入一种新型的神经屏障函数（NBF）来主动检测和过滤不断变化的上下文中出现的有害查询。我们的方法通过学习一个考虑对抗性查询的安全预测器，在每一轮对话中实现不变的安全性，防止潜在的上下文漂移到越狱。在多个LLM下进行的大量实验表明，我们基于NBF的安全转向优于安全对准，基于转向的转向和轻型LLM护栏基线，为多转向越狱提供更强的防御，同时在安全性，有用性和过度拒绝之间保持更好的权衡。查看网站https://sites.google.com/view/llm-nbf/home。我们的代码可以在https://github.com/HanjiangHu/NBF-LLM上找到。



## **22. Stand on The Shoulders of Giants: Building JailExpert from Previous Attack Experience**

站在巨人的肩膀上：建造监狱专家来自之前的攻击经验 cs.CR

18 pages, EMNLP 2025 Main Conference

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.19292v1) [paper-pdf](http://arxiv.org/pdf/2508.19292v1)

**Authors**: Xi Wang, Songlei Jian, Shasha Li, Xiaopeng Li, Bin Ji, Jun Ma, Xiaodong Liu, Jing Wang, Feilong Bao, Jianfeng Zhang, Baosheng Wang, Jie Yu

**Abstract**: Large language models (LLMs) generate human-aligned content under certain safety constraints. However, the current known technique ``jailbreak prompt'' can circumvent safety-aligned measures and induce LLMs to output malicious content. Research on Jailbreaking can help identify vulnerabilities in LLMs and guide the development of robust security frameworks. To circumvent the issue of attack templates becoming obsolete as models evolve, existing methods adopt iterative mutation and dynamic optimization to facilitate more automated jailbreak attacks. However, these methods face two challenges: inefficiency and repetitive optimization, as they overlook the value of past attack experiences. To better integrate past attack experiences to assist current jailbreak attempts, we propose the \textbf{JailExpert}, an automated jailbreak framework, which is the first to achieve a formal representation of experience structure, group experiences based on semantic drift, and support the dynamic updating of the experience pool. Extensive experiments demonstrate that JailExpert significantly improves both attack effectiveness and efficiency. Compared to the current state-of-the-art black-box jailbreak methods, JailExpert achieves an average increase of 17\% in attack success rate and 2.7 times improvement in attack efficiency. Our implementation is available at \href{https://github.com/xiZAIzai/JailExpert}{XiZaiZai/JailExpert}

摘要: 大型语言模型（LLM）在一定的安全约束下生成与人类一致的内容。然而，当前已知的技术“越狱漏洞”可以规避安全性对齐的措施，并诱导LLM输出恶意内容。对越狱的研究可以帮助识别LLM中的漏洞，并指导健壮的安全框架的开发。为了避免攻击模板随着模型的发展而变得过时的问题，现有方法采用迭代变异和动态优化来促进更自动化的越狱攻击。然而，这些方法面临着两个挑战：效率低下和重复优化，因为它们忽视了过去攻击体验的价值。为了更好地整合过去的攻击经验以协助当前的越狱尝试，我们提出了\textBF{JailExpert}，这是一个自动越狱框架，它是第一个实现体验结构的正式表示、基于语义漂移的群组体验，并支持经验池的动态更新。大量实验表明，JailExpert显着提高了攻击有效性和效率。与当前最先进的黑匣子越狱方法相比，JailExpert的攻击成功率平均提高了17%，攻击效率提高了2.7倍。我们的实现可在\href{https：//github.com/xiZAIzai/JailExpert}{XiZaiZai/JailExpert}上获取



## **23. Head-Specific Intervention Can Induce Misaligned AI Coordination in Large Language Models**

特定于头部的干预可能会导致大型语言模型中的AI协调失调 cs.CL

Published at Transaction of Machine Learning Research 08/2025, Large  Language Models (LLMs), Interference-time activation shifting, Steerability,  Explainability, AI alignment, Interpretability

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2502.05945v3) [paper-pdf](http://arxiv.org/pdf/2502.05945v3)

**Authors**: Paul Darm, Annalisa Riccardi

**Abstract**: Robust alignment guardrails for large language models (LLMs) are becoming increasingly important with their widespread application. In contrast to previous studies, we demonstrate that inference-time activation interventions can bypass safety alignments and effectively steer model generations towards harmful AI coordination. Our method applies fine-grained interventions at specific attention heads, which we identify by probing each head in a simple binary choice task. We then show that interventions on these heads generalise to the open-ended generation setting, effectively circumventing safety guardrails. We demonstrate that intervening on a few attention heads is more effective than intervening on full layers or supervised fine-tuning. We further show that only a few example completions are needed to compute effective steering directions, which is an advantage over classical fine-tuning. We also demonstrate that applying interventions in the negative direction can prevent a common jailbreak attack. Our results suggest that, at the attention head level, activations encode fine-grained linearly separable behaviours. Practically, the approach offers a straightforward methodology to steer large language model behaviour, which could be extended to diverse domains beyond safety, requiring fine-grained control over the model output. The code and datasets for this study can be found on https://github.com/PaulDrm/targeted_intervention.

摘要: 随着大型语言模型（LLM）的广泛应用，其稳健的对齐护栏变得越来越重要。与之前的研究相比，我们证明，推理时激活干预可以绕过安全对齐，并有效地引导模型一代转向有害的人工智能协调。我们的方法对特定的注意力头应用细粒度的干预，我们通过在简单的二元选择任务中探测每个注意力头来识别这些注意力头。然后，我们表明，针对这些方面的干预措施普遍适用于开放式的一代环境，有效地绕过了安全护栏。我们证明，干预几个注意头是更有效的比干预全层或监督微调。我们进一步表明，只需要几个例子完成计算有效的转向方向，这是一个优势，经典的微调。我们还证明，在消极的方向上应用干预措施可以防止常见的越狱攻击。我们的研究结果表明，在注意头水平，激活编码细粒度的线性可分离的行为。实际上，该方法提供了一种简单的方法来引导大型语言模型行为，该方法可以扩展到安全以外的不同领域，需要对模型输出进行细粒度控制。本研究的代码和数据集可在https://github.com/PaulDrm/targeted_intervention上找到。



## **24. Speculative Safety-Aware Decoding**

推测性安全意识解码 cs.LG

EMNLP'2025 main conference; more experiments will be added to the  coming camera-ready version

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17739v1) [paper-pdf](http://arxiv.org/pdf/2508.17739v1)

**Authors**: Xuekang Wang, Shengyu Zhu, Xueqi Cheng

**Abstract**: Despite extensive efforts to align Large Language Models (LLMs) with human values and safety rules, jailbreak attacks that exploit certain vulnerabilities continuously emerge, highlighting the need to strengthen existing LLMs with additional safety properties to defend against these attacks. However, tuning large models has become increasingly resource-intensive and may have difficulty ensuring consistent performance. We introduce Speculative Safety-Aware Decoding (SSD), a lightweight decoding-time approach that equips LLMs with the desired safety property while accelerating inference. We assume that there exists a small language model that possesses this desired property. SSD integrates speculative sampling during decoding and leverages the match ratio between the small and composite models to quantify jailbreak risks. This enables SSD to dynamically switch between decoding schemes to prioritize utility or safety, to handle the challenge of different model capacities. The output token is then sampled from a new distribution that combines the distributions of the original and the small models. Experimental results show that SSD successfully equips the large model with the desired safety property, and also allows the model to remain helpful to benign queries. Furthermore, SSD accelerates the inference time, thanks to the speculative sampling design.

摘要: 尽管人们广泛努力将大型语言模型（LLM）与人类价值观和安全规则保持一致，但利用某些漏洞的越狱攻击不断出现，凸显了需要通过额外的安全属性来加强现有的LLM以抵御这些攻击。然而，调整大型模型已变得越来越需要资源密集型，并且可能难以确保一致的性能。我们引入了推测性安全感知解码（SSD），这是一种轻量级解码时间方法，可为LLM配备所需的安全属性，同时加速推理。我们假设存在一个具有这种所需属性的小型语言模型。SSD在解码过程中集成了推测性采样，并利用小模型和复合模型之间的匹配率来量化越狱风险。这使得SSD能够在解码方案之间动态切换，以优先考虑实用性或安全性，以应对不同模型容量的挑战。然后，从结合原始模型和小模型的分布的新分布中对输出令牌进行采样。实验结果表明，SSD成功地装备了大型模型所需的安全属性，也允许模型保持有益的良性查询。此外，由于推测性抽样设计，SSD加快了推理时间。



## **25. Prompt-in-Content Attacks: Exploiting Uploaded Inputs to Hijack LLM Behavior**

内容预算攻击：利用非法输入劫持LLM行为 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.19287v1) [paper-pdf](http://arxiv.org/pdf/2508.19287v1)

**Authors**: Zhuotao Lian, Weiyu Wang, Qingkui Zeng, Toru Nakanishi, Teruaki Kitasuka, Chunhua Su

**Abstract**: Large Language Models (LLMs) are widely deployed in applications that accept user-submitted content, such as uploaded documents or pasted text, for tasks like summarization and question answering. In this paper, we identify a new class of attacks, prompt in content injection, where adversarial instructions are embedded in seemingly benign inputs. When processed by the LLM, these hidden prompts can manipulate outputs without user awareness or system compromise, leading to biased summaries, fabricated claims, or misleading suggestions. We demonstrate the feasibility of such attacks across popular platforms, analyze their root causes including prompt concatenation and insufficient input isolation, and discuss mitigation strategies. Our findings reveal a subtle yet practical threat in real-world LLM workflows.

摘要: 大型语言模型（LLM）广泛部署在接受用户提交的内容（例如上传的文档或粘贴的文本）的应用程序中，以执行总结和问答等任务。在本文中，我们识别了一类新的攻击，在内容注入中提示，其中对抗性指令被嵌入到看似良性的输入中。当LLM处理时，这些隐藏的提示可能会在用户意识不到或系统损害的情况下操纵输出，从而导致有偏见的摘要、捏造的主张或误导性的建议。我们展示了此类攻击在流行平台上的可行性，分析了其根本原因，包括迅速级联和输入隔离不足，并讨论了缓解策略。我们的研究结果揭示了现实世界LLM工作流程中一个微妙但实际的威胁。



## **26. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

攻击LLM和AI代理：针对大型语言模型的广告嵌入攻击 cs.CR

7 pages, 2 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17674v1) [paper-pdf](http://arxiv.org/pdf/2508.17674v1)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.

摘要: 我们引入了广告嵌入攻击（AEA），这是一种新型LLM安全威胁，可以悄悄地将促销或恶意内容注入模型输出和AI代理中。AEA通过两种低成本载体运作：（1）劫持第三方服务分发平台以预先设置对抗提示，以及（2）发布经过攻击者数据微调的后门开源检查点。与降低准确性的传统攻击不同，AEA破坏了信息完整性，导致模型在看起来正常的情况下返回秘密广告、宣传或仇恨言论。我们详细介绍了攻击管道，绘制了五个利益相关者受害者群体，并提出了一种初步的基于预算的自我检查防御，该防御可以减轻这些注入，而无需额外的模型再培训。我们的调查结果揭示了LLM安全方面存在一个紧迫且未充分解决的差距，并呼吁人工智能安全界协调检测、审计和政策响应。



## **27. TombRaider: Entering the Vault of History to Jailbreak Large Language Models**

TombRaider：进入历史宝库越狱大型语言模型 cs.CR

Main Conference of EMNLP

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2501.18628v2) [paper-pdf](http://arxiv.org/pdf/2501.18628v2)

**Authors**: Junchen Ding, Jiahao Zhang, Yi Liu, Ziqi Ding, Gelei Deng, Yuekang Li

**Abstract**: Warning: This paper contains content that may involve potentially harmful behaviours, discussed strictly for research purposes.   Jailbreak attacks can hinder the safety of Large Language Model (LLM) applications, especially chatbots. Studying jailbreak techniques is an important AI red teaming task for improving the safety of these applications. In this paper, we introduce TombRaider, a novel jailbreak technique that exploits the ability to store, retrieve, and use historical knowledge of LLMs. TombRaider employs two agents, the inspector agent to extract relevant historical information and the attacker agent to generate adversarial prompts, enabling effective bypassing of safety filters. We intensively evaluated TombRaider on six popular models. Experimental results showed that TombRaider could outperform state-of-the-art jailbreak techniques, achieving nearly 100% attack success rates (ASRs) on bare models and maintaining over 55.4% ASR against defence mechanisms. Our findings highlight critical vulnerabilities in existing LLM safeguards, underscoring the need for more robust safety defences.

摘要: 警告：本文包含可能涉及潜在有害行为的内容，严格出于研究目的进行讨论。   越狱攻击可能会阻碍大型语言模型（LLM）应用程序的安全性，尤其是聊天机器人。研究越狱技术是提高这些应用安全性的一项重要人工智能红色团队任务。本文中，我们介绍了TombRaider，这是一种新型越狱技术，它利用了存储、检索和使用LLM历史知识的能力。TombRaider使用两个代理，检查员代理提取相关历史信息，攻击者代理生成对抗提示，从而有效绕过安全过滤器。我们对TombRaider的六款热门型号进行了深入评估。实验结果表明，TombRaider的性能优于最先进的越狱技术，在裸模型上实现了近100%的攻击成功率（ASB），并在防御机制下保持超过55.4%的ASB。我们的调查结果强调了现有LLM保障措施中的关键漏洞，强调了更强大的安全防御的必要性。



## **28. RL-Finetuned LLMs for Privacy-Preserving Synthetic Rewriting**

RL微调LLM用于隐私保护综合重写 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.19286v1) [paper-pdf](http://arxiv.org/pdf/2508.19286v1)

**Authors**: Zhan Shi, Yefeng Yuan, Yuhong Liu, Liang Cheng, Yi Fang

**Abstract**: The performance of modern machine learning systems depends on access to large, high-quality datasets, often sourced from user-generated content or proprietary, domain-specific corpora. However, these rich datasets inherently contain sensitive personal information, raising significant concerns about privacy, data security, and compliance with regulatory frameworks. While conventional anonymization techniques can remove explicit identifiers, such removal may result in performance drop in downstream machine learning tasks. More importantly, simple anonymization may not be effective against inference attacks that exploit implicit signals such as writing style, topical focus, or demographic cues, highlighting the need for more robust privacy safeguards during model training. To address the challenging issue of balancing user privacy and data utility, we propose a reinforcement learning framework that fine-tunes a large language model (LLM) using a composite reward function that jointly optimizes for explicit and implicit privacy, semantic fidelity, and output diversity. To effectively capture population level regularities, the privacy reward combines semantic cues with structural patterns derived from a minimum spanning tree (MST) over latent representations. By modeling these privacy-sensitive signals in their distributional context, the proposed approach guides the model to generate synthetic rewrites that preserve utility while mitigating privacy risks. Empirical results show that the proposed method significantly enhances author obfuscation and privacy metrics without degrading semantic quality, providing a scalable and model-agnostic solution for privacy preserving data generation in the era of large language models.

摘要: 现代机器学习系统的性能取决于对大型、高质量数据集的访问，这些数据集通常来自用户生成的内容或专有的、特定领域的文集。然而，这些丰富的数据集本质上包含敏感的个人信息，引发了人们对隐私、数据安全和监管框架合规性的严重担忧。虽然传统的匿名化技术可以删除显式标识符，但这种删除可能会导致下游机器学习任务的性能下降。更重要的是，简单的匿名化可能无法有效地对抗利用写作风格、话题焦点或人口统计线索等隐性信号的推理攻击，这凸显了模型训练期间需要更强大的隐私保护措施。为了解决平衡用户隐私和数据效用的挑战性问题，我们提出了一种强化学习框架，该框架使用复合奖励函数来微调大型语言模型（LLM），该函数联合优化显式和隐式隐私、语义保真度和输出多样性。为了有效地捕获人口水平的预设，隐私奖励将语义线索与从潜在表示上的最小生成树（MST）推导出的结构模式相结合。通过在分布环境中对这些隐私敏感的信号进行建模，提出的方法引导模型生成合成重写，以保留效用，同时降低隐私风险。实验结果表明，所提出的方法在不降低语义质量的情况下显着增强了作者混淆和隐私指标，为大型语言模型时代的隐私保护数据生成提供了可扩展且模型不可知的解决方案。



## **29. Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models**

自适应语言过滤（ALP）增强多模态大型语言模型中的网络钓鱼网页检测 cs.CL

Published at ACL 2025 SRW, 9 pages, 3 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.13357v2) [paper-pdf](http://arxiv.org/pdf/2507.13357v2)

**Authors**: Atharva Bhargude, Ishan Gonehal, Dave Yoon, Kaustubh Vinnakota, Chandler Haney, Aaron Sandoval, Kevin Zhu

**Abstract**: Phishing attacks represent a significant cybersecurity threat, necessitating adaptive detection techniques. This study explores few-shot Adaptive Linguistic Prompting (ALP) in detecting phishing webpages through the multimodal capabilities of state-of-the-art large language models (LLMs) such as GPT-4o and Gemini 1.5 Pro. ALP is a structured semantic reasoning method that guides LLMs to analyze textual deception by breaking down linguistic patterns, detecting urgency cues, and identifying manipulative diction commonly found in phishing content. By integrating textual, visual, and URL-based analysis, we propose a unified model capable of identifying sophisticated phishing attempts. Our experiments demonstrate that ALP significantly enhances phishing detection accuracy by guiding LLMs through structured reasoning and contextual analysis. The findings highlight the potential of ALP-integrated multimodal LLMs to advance phishing detection frameworks, achieving an F1-score of 0.93, surpassing traditional approaches. These results establish a foundation for more robust, interpretable, and adaptive linguistic-based phishing detection systems using LLMs.

摘要: 网络钓鱼攻击是一个重大的网络安全威胁，需要自适应检测技术。本研究探索了通过GPT-4 o和Gemini 1.5 Pro等最先进大型语言模型（LLM）的多模式功能检测网络钓鱼网页的几次自适应语言预测（AFP）。ALA是一种结构化语义推理方法，通过分解语言模式、检测紧迫性线索和识别网络钓鱼内容中常见的操纵性措辞来指导LLM分析文本欺骗。通过集成文本、视觉和基于URL的分析，我们提出了一个能够识别复杂的网络钓鱼企图的统一模型。我们的实验表明，通过结构化推理和上下文分析来指导LLM，ALA显着提高了网络钓鱼检测的准确性。研究结果凸显了整合了AFP的多模式LLM在推进网络钓鱼检测框架方面的潜力，F1评分达到0.93，超越了传统方法。这些结果为使用LLM的更稳健、可解释和自适应的基于语言的网络钓鱼检测系统奠定了基础。



## **30. Unified attacks to large language model watermarks: spoofing and scrubbing in unauthorized knowledge distillation**

对大型语言模型水印的统一攻击：未经授权的知识提炼中的欺骗和擦洗 cs.CL

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2504.17480v4) [paper-pdf](http://arxiv.org/pdf/2504.17480v4)

**Authors**: Xin Yi, Yue Li, Shunfan Zheng, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable.

摘要: 水印已成为打击错误信息和保护大型语言模型（LLM）知识产权的关键技术。最近的一项发现称为水印放射性，揭示了教师模型中嵌入的水印可以通过知识蒸馏被学生模型继承。从积极的方面来说，这种继承允许通过识别学生模型中的水印痕迹来检测未经授权的知识提炼。然而，水印对擦洗攻击的鲁棒性及其在未经授权的知识提炼下面对欺骗攻击时的不可伪造性在很大程度上仍然没有被探索。现有的水印攻击方法要么假设访问模型内部内容，要么无法同时支持擦洗和欺骗攻击。在这项工作中，我们提出了对比解码引导知识蒸馏（CDG-KD），这是一个统一框架，可以在未经授权的知识蒸馏下实现双向攻击。我们的方法采用对比解码，通过比较学生模型和弱水印参考的输出来提取损坏或放大的水印文本，然后进行双向蒸馏以分别训练能够去除水印和伪造水印的新学生模型。大量实验表明，CDG-KD可以有效地执行攻击，同时保持提取模型的一般性能。我们的研究结果强调了开发稳健且不可伪造的水印方案的迫切需要。



## **31. Defending against Jailbreak through Early Exit Generation of Large Language Models**

通过早期退出生成大型语言模型抵御越狱 cs.AI

ICONIP 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2408.11308v2) [paper-pdf](http://arxiv.org/pdf/2408.11308v2)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. We introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85% in comparison with 50% for the present SOTAs, with minimal impact on the utility of LLMs.

摘要: 大型语言模型（LLM）在各种应用中越来越受到关注。尽管如此，随着一些用户试图利用这些模型进行恶意目的，包括合成受控物质和传播虚假信息，人们越来越担心。为了降低此类风险，“对齐”技术的概念被开发出来。然而，最近的研究表明，使用复杂的即时工程或对抗性后缀（一种被称为“越狱”的技术）可能会破坏这种对齐。“我们的研究从LLM的类人类生成过程中汲取线索。我们发现，虽然越狱提示可能会产生类似于良性提示的输出日志，但它们在模型潜在空间中的初始嵌入往往更类似于恶意提示的嵌入。利用这一发现，我们建议利用LLM的早期Transformer输出作为检测恶意输入并立即终止生成的手段。我们为LLM引入了一种简单但重要的防御方法，称为EEG-Defender。我们对三种模型的十种越狱方法进行了全面实验。我们的结果表明，EEG-Defender能够大幅降低攻击成功率（ASB），大约为85%，而当前SOTA的攻击成功率为50%，对LLM的实用性影响最小。



## **32. Exploring the Vulnerability of the Content Moderation Guardrail in Large Language Models via Intent Manipulation**

通过意图操纵探索大型语言模型中内容审核保护的脆弱性 cs.CL

Accepted for EMNLP'25 Findings. TL;DR: We propose a new two-stage  intent-based prompt-refinement framework, IntentPrompt, that aims to explore  the vulnerability of LLMs' content moderation guardrails by refining prompts  into benign-looking declarative forms via intent manipulation for red-teaming  purposes

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2505.18556v2) [paper-pdf](http://arxiv.org/pdf/2505.18556v2)

**Authors**: Jun Zhuang, Haibo Jin, Ye Zhang, Zhengjian Kang, Wenbin Zhang, Gaby G. Dagher, Haohan Wang

**Abstract**: Intent detection, a core component of natural language understanding, has considerably evolved as a crucial mechanism in safeguarding large language models (LLMs). While prior work has applied intent detection to enhance LLMs' moderation guardrails, showing a significant success against content-level jailbreaks, the robustness of these intent-aware guardrails under malicious manipulations remains under-explored. In this work, we investigate the vulnerability of intent-aware guardrails and demonstrate that LLMs exhibit implicit intent detection capabilities. We propose a two-stage intent-based prompt-refinement framework, IntentPrompt, that first transforms harmful inquiries into structured outlines and further reframes them into declarative-style narratives by iteratively optimizing prompts via feedback loops to enhance jailbreak success for red-teaming purposes. Extensive experiments across four public benchmarks and various black-box LLMs indicate that our framework consistently outperforms several cutting-edge jailbreak methods and evades even advanced Intent Analysis (IA) and Chain-of-Thought (CoT)-based defenses. Specifically, our "FSTR+SPIN" variant achieves attack success rates ranging from 88.25% to 96.54% against CoT-based defenses on the o1 model, and from 86.75% to 97.12% on the GPT-4o model under IA-based defenses. These findings highlight a critical weakness in LLMs' safety mechanisms and suggest that intent manipulation poses a growing challenge to content moderation guardrails.

摘要: 意图检测是自然语言理解的核心组成部分，已经发展成为保护大型语言模型（LLM）的关键机制。虽然先前的工作已经应用意图检测来增强LLM的适度护栏，显示出对内容级越狱的显著成功，但这些意图感知护栏在恶意操纵下的鲁棒性仍然未被充分探索。在这项工作中，我们调查意图感知护栏的脆弱性，并证明LLM表现出隐式意图检测能力。我们提出了一个两阶段的基于意图的越狱细化框架，IntentPrompt，首先将有害的查询转换为结构化的大纲，并通过反馈循环迭代优化提示，以提高越狱成功率，从而进一步将其重新构建为声明式的叙述。针对四个公共基准测试和各种黑匣子LLM的广泛实验表明，我们的框架始终优于几种尖端的越狱方法，甚至可以规避高级意图分析（IA）和基于思想链（CoT）的防御。具体来说，我们的“FTR +SPIN”变体在o 1模型上针对基于CoT的防御的攻击成功率从88.25%到96.54%不等，在基于IA的防御下，在GPT-4 o模型上的攻击成功率从86.75%到97.12%不等。这些发现凸显了LLM安全机制的一个严重弱点，并表明意图操纵对内容审核护栏构成了越来越大的挑战。



## **33. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

相信我，我知道这个功能：使用偏差劫持LLM静态分析 cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17361v1) [paper-pdf](http://arxiv.org/pdf/2508.17361v1)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective, but also transferable across models (GPT-4o, Claude 3.5, Gemini 2.0) and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.

摘要: 大型语言模型（LLM）越来越被信任大规模执行自动代码审查和静态分析，支持漏洞检测、总结和重构等任务。在本文中，我们识别并利用基于LLM的代码分析中的一个关键漏洞：一种抽象偏见，导致模型过度概括熟悉的编程模式并忽略小而有意义的错误。对手可以利用这个盲点来劫持LLM解释的控制流，只需最少的编辑，并且不会影响实际的运行时行为。我们将这种攻击称为熟悉模式攻击（FTA）。   我们开发了一种全自动的黑匣子算法，可以发现并将FPA注入目标代码。我们的评估表明，PFA不仅有效，而且可以跨模型（GPT-4 o、Claude 3.5、Gemini 2.0）移植，并且跨编程语言（Python、C、Rust、Go）通用。此外，即使通过强大的系统提示明确警告模型有关攻击，FPA仍然有效。最后，我们探讨了PFA的积极、防御性用途，并讨论了它们对面向代码的LLM的可靠性和安全性的更广泛影响。



## **34. Risk Assessment and Security Analysis of Large Language Models**

大型语言模型的风险评估与安全性分析 cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17329v1) [paper-pdf](http://arxiv.org/pdf/2508.17329v1)

**Authors**: Xiaoyan Zhang, Dongyang Lyu, Xiaoqi Li

**Abstract**: As large language models (LLMs) expose systemic security challenges in high risk applications, including privacy leaks, bias amplification, and malicious abuse, there is an urgent need for a dynamic risk assessment and collaborative defence framework that covers their entire life cycle. This paper focuses on the security problems of large language models (LLMs) in critical application scenarios, such as the possibility of disclosure of user data, the deliberate input of harmful instructions, or the models bias. To solve these problems, we describe the design of a system for dynamic risk assessment and a hierarchical defence system that allows different levels of protection to cooperate. This paper presents a risk assessment system capable of evaluating both static and dynamic indicators simultaneously. It uses entropy weighting to calculate essential data, such as the frequency of sensitive words, whether the API call is typical, the realtime risk entropy value is significant, and the degree of context deviation. The experimental results show that the system is capable of identifying concealed attacks, such as role escape, and can perform rapid risk evaluation. The paper uses a hybrid model called BERT-CRF (Bidirectional Encoder Representation from Transformers) at the input layer to identify and filter malicious commands. The model layer uses dynamic adversarial training and differential privacy noise injection technology together. The output layer also has a neural watermarking system that can track the source of the content. In practice, the quality of this method, especially important in terms of customer service in the financial industry.

摘要: 由于大型语言模型（LLM）在高风险应用中暴露出系统性的安全挑战，包括隐私泄露，偏见放大和恶意滥用，因此迫切需要一个涵盖其整个生命周期的动态风险评估和协作防御框架。本文重点研究了大型语言模型在关键应用场景中的安全问题，如用户数据泄露的可能性、有害指令的故意输入、模型偏差等。为了解决这些问题，我们描述了一个系统的设计，动态风险评估和分级防御系统，允许不同级别的保护合作。本文提出了一种能够同时评估静态和动态指标的风险评估系统。它使用熵加权来计算基本数据，例如敏感词的频率、API调用是否典型、实时风险熵值是否重要以及上下文偏离程度。实验结果表明，该系统能够识别角色逃避等隐藏攻击，并能够进行快速风险评估。该论文在输入层使用了一种名为BERT-RF（来自Transformers的双向编码器表示）的混合模型来识别和过滤恶意命令。模型层结合使用动态对抗训练和差异隐私噪音注入技术。输出层还具有一个可以跟踪内容来源的神经水印系统。在实践中，这种方法的质量对于金融行业的客户服务尤其重要。



## **35. Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks**

具有免训练连续投影的细粒度安全神经元，以降低LLM微调风险 cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.09190v3) [paper-pdf](http://arxiv.org/pdf/2508.09190v3)

**Authors**: Bing Han, Feifei Zhao, Dongcheng Zhao, Guobin Shen, Ping Wu, Yu Shi, Yi Zeng

**Abstract**: Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns.

摘要: 微调即服务将特定领域的知识注入到大型语言模型（LLM）中，同时挑战了原始的对齐机制并引入了安全风险。针对对齐、微调和微调后阶段提出了一系列防御策略，其中大多数微调后防御依赖于粗粒度安全层映射。这些方法缺乏对安全层和细粒度神经元的综合考虑，限制了它们有效平衡安全性和实用性的能力。为了解决这个问题，我们提出了细粒度安全神经元（FGSN）与训练免费连续投影方法，以减少微调的安全风险。FGSN固有地集成了安全层和神经元之间的多尺度交互，定位更稀疏和更精确的细粒度安全神经元，同时最大限度地减少对下游任务神经元的干扰。然后，我们将安全神经元参数投影到安全方向上，提高模型的安全性，同时更紧密地与人类偏好保持一致。在多个微调的LLM模型上进行的广泛实验表明，我们的方法在保持模型实用性的同时，以最小的参数修改显着降低了危害分数和攻击成功率。此外，通过引入特定于任务的多维异构安全神经元簇优化机制，我们实现了对不可预见的新出现的安全问题的持续防御和泛化能力。



## **36. Exposing Privacy Risks in Graph Retrieval-Augmented Generation**

暴露图形检索增强一代中的隐私风险 cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17222v1) [paper-pdf](http://arxiv.org/pdf/2508.17222v1)

**Authors**: Jiale Liu, Jiahao Zhang, Suhang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) is a powerful technique for enhancing Large Language Models (LLMs) with external, up-to-date knowledge. Graph RAG has emerged as an advanced paradigm that leverages graph-based knowledge structures to provide more coherent and contextually rich answers. However, the move from plain document retrieval to structured graph traversal introduces new, under-explored privacy risks. This paper investigates the data extraction vulnerabilities of the Graph RAG systems. We design and execute tailored data extraction attacks to probe their susceptibility to leaking both raw text and structured data, such as entities and their relationships. Our findings reveal a critical trade-off: while Graph RAG systems may reduce raw text leakage, they are significantly more vulnerable to the extraction of structured entity and relationship information. We also explore potential defense mechanisms to mitigate these novel attack surfaces. This work provides a foundational analysis of the unique privacy challenges in Graph RAG and offers insights for building more secure systems.

摘要: 检索增强生成（RAG）是一种利用外部最新知识增强大型语言模型（LLM）的强大技术。图RAG已成为一种先进的范式，它利用基于图的知识结构来提供更连贯且上下文丰富的答案。然而，从普通文档检索到结构化图穿越的转变引入了新的、未充分探索的隐私风险。本文研究了Shape RAG系统的数据提取漏洞。我们设计并执行量身定制的数据提取攻击，以调查它们对泄露原始文本和结构化数据（例如实体及其关系）的敏感性。我们的研究结果揭示了一个关键的权衡：虽然Shape RAG系统可能会减少原始文本泄露，但它们明显更容易受到结构化实体和关系信息的提取的影响。我们还探索潜在的防御机制来减轻这些新型攻击表面。这项工作为Shape RAG中独特的隐私挑战提供了基础分析，并为构建更安全的系统提供了见解。



## **37. How to make Medical AI Systems safer? Simulating Vulnerabilities, and Threats in Multimodal Medical RAG System**

如何让医疗人工智能系统更安全？模拟多模式医疗RAG系统中的漏洞和威胁 cs.LG

Sumbitted to 2025 AAAI main track

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17215v1) [paper-pdf](http://arxiv.org/pdf/2508.17215v1)

**Authors**: Kaiwen Zuo, Zelin Liu, Raman Dutt, Ziyang Wang, Zhongtian Sun, Yeming Wang, Fan Mo, Pietro Liò

**Abstract**: Large Vision-Language Models (LVLMs) augmented with Retrieval-Augmented Generation (RAG) are increasingly employed in medical AI to enhance factual grounding through external clinical image-text retrieval. However, this reliance creates a significant attack surface. We propose MedThreatRAG, a novel multimodal poisoning framework that systematically probes vulnerabilities in medical RAG systems by injecting adversarial image-text pairs. A key innovation of our approach is the construction of a simulated semi-open attack environment, mimicking real-world medical systems that permit periodic knowledge base updates via user or pipeline contributions. Within this setting, we introduce and emphasize Cross-Modal Conflict Injection (CMCI), which embeds subtle semantic contradictions between medical images and their paired reports. These mismatches degrade retrieval and generation by disrupting cross-modal alignment while remaining sufficiently plausible to evade conventional filters. While basic textual and visual attacks are included for completeness, CMCI demonstrates the most severe degradation. Evaluations on IU-Xray and MIMIC-CXR QA tasks show that MedThreatRAG reduces answer F1 scores by up to 27.66% and lowers LLaVA-Med-1.5 F1 rates to as low as 51.36%. Our findings expose fundamental security gaps in clinical RAG systems and highlight the urgent need for threat-aware design and robust multimodal consistency checks. Finally, we conclude with a concise set of guidelines to inform the safe development of future multimodal medical RAG systems.

摘要: 通过检索增强生成（RAG）增强的大型视觉语言模型（LVLM）越来越多地用于医疗人工智能，以通过外部临床图像文本检索增强事实基础。然而，这种依赖造成了重大的攻击面。我们提出了MedThreatRAG，这是一种新型的多模式中毒框架，通过注入对抗性图像-文本对来系统性地探索医疗RAG系统中的漏洞。我们方法的一个关键创新是构建模拟的半开放攻击环境，模仿现实世界的医疗系统，允许通过用户或管道贡献定期更新知识库。在此背景下，我们引入并强调跨模式冲突注入（CCGI），它嵌入了医学图像及其配对报告之间的微妙语义矛盾。这些不匹配通过扰乱跨模式对齐而降低检索和生成，同时保持足够合理以逃避传统过滤器。虽然为了完整性而包含了基本的文本和视觉攻击，但CMCI表现出了最严重的降级。对IU-X射线和MIIC-CXR QA任务的评估表明，MedThreatRAG将答案F1评分降低高达27.66%，并将LLaBA-Med-1.5 F1评分降低至低至51.36%。我们的研究结果揭示了临床RAG系统中的根本安全漏洞，并强调了对威胁感知设计和强大的多模式一致性检查的迫切需求。最后，我们提出了一套简洁的指南，为未来多模式医疗RAG系统的安全开发提供信息。



## **38. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

对LLM as-a-Judge的基于优化的即时注入攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2403.17710v5) [paper-pdf](http://arxiv.org/pdf/2403.17710v5)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies. Our implementation is available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.

摘要: LLM作为法官使用大型语言模型（LLM）从给定问题的一组候选者中选择最佳答案。LLM as-a-Judge具有许多应用，例如LLM驱动的搜索、具有人工智能反馈的强化学习（RLAIF）和工具选择。在这项工作中，我们提出了JudggeDeceiver，这是一种针对LLM as-a-Judge的基于优化的即时注入攻击。JudggeDeceiver将精心制作的序列注入到攻击者控制的候选人响应中，以便LLM法官都会为攻击者选择的问题选择候选人响应，无论其他候选人响应是什么。具体来说，我们将寻找这样的序列作为一个优化问题，并提出了一种基于梯度的方法来近似解决它。我们的广泛评估表明，JudggeDeceive非常有效，并且比现有的手动制作注入序列的即时注入攻击和扩展到我们的问题时的越狱攻击有效得多。我们还在三个案例研究中展示了JudgeDeceiver的有效性，即LLM支持的搜索、RLAIF和工具选择。此外，我们还考虑了包括已知答案检测、困惑度检测和困惑度窗口检测在内的防御措施。我们的结果表明这些防御措施还不够，凸显了开发新防御策略的迫切需要。我们的实现可在此存储库中获取：https://github.com/ShiJiawenwen/JudgeDeceiver。



## **39. Security Concerns for Large Language Models: A Survey**

大型语言模型的安全问题：调查 cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2505.18889v5) [paper-pdf](http://arxiv.org/pdf/2505.18889v5)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as ChatGPT and its competitors have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. This survey provides a comprehensive overview of these emerging concerns, categorizing threats into several key areas: inference-time attacks via prompt manipulation; training-time attacks; misuse by malicious actors; and the inherent risks in autonomous LLM agents. Recently, a significant focus is increasingly being placed on the latter. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze existing defense mechanisms and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.

摘要: ChatGPT及其竞争对手等大型语言模型（LLM）引发了自然语言处理领域的一场革命，但它们的功能也引入了新的安全漏洞。本调查全面概述了这些新出现的问题，将威胁分为几个关键领域：通过即时操纵进行的推理时间攻击;训练时间攻击;恶意行为者的滥用;以及自主LLM代理的固有风险。最近，后者越来越受到人们的关注。我们总结了2022年至2025年期间最近的学术和工业研究，这些研究揭示了每种威胁，分析了现有的防御机制及其局限性，并确定了保护基于LLM的应用程序方面的公开挑战。最后，我们强调了推进强大的多层安全策略以确保LLM安全且有益的重要性。



## **40. Towards Safeguarding LLM Fine-tuning APIs against Cipher Attacks**

保护LLM微调API免受密码攻击 cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.17158v1) [paper-pdf](http://arxiv.org/pdf/2508.17158v1)

**Authors**: Jack Youstra, Mohammed Mahfoud, Yang Yan, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: Large language model fine-tuning APIs enable widespread model customization, yet pose significant safety risks. Recent work shows that adversaries can exploit access to these APIs to bypass model safety mechanisms by encoding harmful content in seemingly harmless fine-tuning data, evading both human monitoring and standard content filters. We formalize the fine-tuning API defense problem, and introduce the Cipher Fine-tuning Robustness benchmark (CIFR), a benchmark for evaluating defense strategies' ability to retain model safety in the face of cipher-enabled attackers while achieving the desired level of fine-tuning functionality. We include diverse cipher encodings and families, with some kept exclusively in the test set to evaluate for generalization across unseen ciphers and cipher families. We then evaluate different defenses on the benchmark and train probe monitors on model internal activations from multiple fine-tunes. We show that probe monitors achieve over 99% detection accuracy, generalize to unseen cipher variants and families, and compare favorably to state-of-the-art monitoring approaches. We open-source CIFR and the code to reproduce our experiments to facilitate further research in this critical area. Code and data are available online https://github.com/JackYoustra/safe-finetuning-api

摘要: 大型语言模型微调API可以实现广泛的模型定制，但也会带来重大的安全风险。最近的工作表明，对手可以利用对这些API的访问来绕过模型安全机制，将有害内容编码在看似无害的微调数据中，从而逃避人类监控和标准内容过滤器。我们形式化了微调API防御问题，并引入了Cipher微调稳健性基准（CIFR），这是一个评估防御策略在面对启用密码的攻击者时保持模型安全性的能力的基准，同时实现了所需的微调功能水平。我们包括不同的密码编码和系列，其中一些仅保留在测试集中，以评估未见密码和密码系列的通用性。然后，我们评估基准上的不同防御，并根据多个微调的模型内部激活训练探测器监视器。我们表明，探针监测器实现了超过99%的检测准确率，可推广到未见的密码变体和家族，并且与最先进的监测方法相比具有优势。我们开源CIFR和复制我们实验的代码，以促进这一关键领域的进一步研究。代码和数据可在线获取https://github.com/JackYoustra/safe-finetuning-api



## **41. Mind the Gap: Time-of-Check to Time-of-Use Vulnerabilities in LLM-Enabled Agents**

注意差距：支持LLM的代理中的检查时间到使用时间漏洞 cs.CR

Pre-print

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.17155v1) [paper-pdf](http://arxiv.org/pdf/2508.17155v1)

**Authors**: Derek Lilienthal, Sanghyun Hong

**Abstract**: Large Language Model (LLM)-enabled agents are rapidly emerging across a wide range of applications, but their deployment introduces vulnerabilities with security implications. While prior work has examined prompt-based attacks (e.g., prompt injection) and data-oriented threats (e.g., data exfiltration), time-of-check to time-of-use (TOCTOU) remain largely unexplored in this context. TOCTOU arises when an agent validates external state (e.g., a file or API response) that is later modified before use, enabling practical attacks such as malicious configuration swaps or payload injection. In this work, we present the first study of TOCTOU vulnerabilities in LLM-enabled agents. We introduce TOCTOU-Bench, a benchmark with 66 realistic user tasks designed to evaluate this class of vulnerabilities. As countermeasures, we adapt detection and mitigation techniques from systems security to this setting and propose prompt rewriting, state integrity monitoring, and tool-fusing. Our study highlights challenges unique to agentic workflows, where we achieve up to 25% detection accuracy using automated detection methods, a 3% decrease in vulnerable plan generation, and a 95% reduction in the attack window. When combining all three approaches, we reduce the TOCTOU vulnerabilities from an executed trajectory from 12% to 8%. Our findings open a new research direction at the intersection of AI safety and systems security.

摘要: 支持大型语言模型（LLM）的代理正在广泛的应用程序中迅速出现，但它们的部署会引入具有安全影响的漏洞。虽然之前的工作已经研究了基于预算的攻击（例如，即时注入）和面向数据的威胁（例如，数据外流）、检查时间到使用时间（TOCTSYS）在这种背景下基本上仍未被探索。当代理验证外部状态（例如，文件或API响应），稍后在使用前进行修改，从而实现恶意配置交换或有效负载注入等实际攻击。在这项工作中，我们首次研究了LLM启用的代理中的TOCTSYS漏洞。我们引入了TOCTOU-Bench，这是一个具有66个现实用户任务的基准，旨在评估此类漏洞。作为应对措施，我们将系统安全的检测和缓解技术调整到这种设置，并提出即时重写、状态完整性监控和工具融合。我们的研究强调了代理工作流程所独有的挑战，我们使用自动检测方法实现了高达25%的检测准确率，脆弱计划生成减少3%，攻击窗口减少95%。当结合这三种方法时，我们将执行轨迹中的TOCTSYS漏洞从12%减少到8%。我们的研究结果为人工智能安全和系统安全的交叉点开辟了新的研究方向。



## **42. POT: Inducing Overthinking in LLMs via Black-Box Iterative Optimization**

POT：通过黑匣子迭代优化在LLM中引发过度思考 cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.19277v1) [paper-pdf](http://arxiv.org/pdf/2508.19277v1)

**Authors**: Xinyu Li, Tianjin Huang, Ronghui Mu, Xiaowei Huang, Gaojie Jin

**Abstract**: Recent advances in Chain-of-Thought (CoT) prompting have substantially enhanced the reasoning capabilities of large language models (LLMs), enabling sophisticated problem-solving through explicit multi-step reasoning traces. However, these enhanced reasoning processes introduce novel attack surfaces, particularly vulnerabilities to computational inefficiency through unnecessarily verbose reasoning chains that consume excessive resources without corresponding performance gains. Prior overthinking attacks typically require restrictive conditions including access to external knowledge sources for data poisoning, reliance on retrievable poisoned content, and structurally obvious templates that limit practical applicability in real-world scenarios. To address these limitations, we propose POT (Prompt-Only OverThinking), a novel black-box attack framework that employs LLM-based iterative optimization to generate covert and semantically natural adversarial prompts, eliminating dependence on external data access and model retrieval. Extensive experiments across diverse model architectures and datasets demonstrate that POT achieves superior performance compared to other methods.

摘要: 思想链（CoT）提示的最新进展大大增强了大型语言模型（LLM）的推理能力，通过显式的多步骤推理痕迹实现复杂的问题解决。然而，这些增强的推理过程引入了新颖的攻击表面，特别是由于不必要的冗长推理链而导致计算效率低下的脆弱性，这些推理链消耗了过多的资源而没有相应的性能提升。之前的过度思考攻击通常需要限制性条件，包括访问外部知识源进行数据中毒、依赖可检索的中毒内容以及限制现实世界场景中实际适用性的结构明显模板。为了解决这些限制，我们提出了POT（仅预算过度思考），这是一种新型黑匣子攻击框架，它采用基于LLM的迭代优化来生成隐蔽且语义自然的对抗提示，消除了对外部数据访问和模型检索的依赖。跨不同模型架构和数据集的广泛实验表明，与其他方法相比，POT实现了更卓越的性能。



## **43. Unveiling the Latent Directions of Reflection in Large Language Models**

揭示大型语言模型中反射的潜在方向 cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.16989v1) [paper-pdf](http://arxiv.org/pdf/2508.16989v1)

**Authors**: Fu-Chieh Chang, Yu-Ting Lee, Pei-Yuan Wu

**Abstract**: Reflection, the ability of large language models (LLMs) to evaluate and revise their own reasoning, has been widely used to improve performance on complex reasoning tasks. Yet, most prior work emphasizes designing reflective prompting strategies or reinforcement learning objectives, leaving the inner mechanisms of reflection underexplored. In this paper, we investigate reflection through the lens of latent directions in model activations. We propose a methodology based on activation steering to characterize how instructions with different reflective intentions: no reflection, intrinsic reflection, and triggered reflection. By constructing steering vectors between these reflection levels, we demonstrate that (1) new reflection-inducing instructions can be systematically identified, (2) reflective behavior can be directly enhanced or suppressed through activation interventions, and (3) suppressing reflection is considerably easier than stimulating it. Experiments on GSM8k-adv with Qwen2.5-3B and Gemma3-4B reveal clear stratification across reflection levels, and steering interventions confirm the controllability of reflection. Our findings highlight both opportunities (e.g., reflection-enhancing defenses) and risks (e.g., adversarial inhibition of reflection in jailbreak attacks). This work opens a path toward mechanistic understanding of reflective reasoning in LLMs.

摘要: 反射是大型语言模型（LLM）评估和修改自身推理的能力，已被广泛用于提高复杂推理任务的性能。然而，大多数先前的工作都强调设计反思性提示策略或强化学习目标，而反思的内部机制却没有得到充分的探索。本文中，我们研究了模型激活中潜在方向的镜头的反射。我们提出了一种基于激活引导的方法论来描述具有不同反射意图的指令：无反射、内在反射和触发反射。通过在这些反射水平之间构建引导载体，我们证明了（1）可以系统地识别新的反射诱导指令，（2）可以通过激活干预直接增强或抑制反射行为，（3）抑制反射比刺激反射容易得多。在GSM 8 k-adv上使用Qwen 2.5 -3B和Gemma 3 - 4 B进行的实验揭示了反射水平之间的明显分层，而引导干预证实了反思的可控性。我们的调查结果强调了这两种机会（例如，反思增强防御）和风险（例如，越狱攻击中反思的对抗性抑制）。这项工作开辟了对法学硕士中反思推理的机械理解的道路。



## **44. Mitigating Jailbreaks with Intent-Aware LLMs**

利用意图意识的法学硕士缓解越狱 cs.CR

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.12072v2) [paper-pdf](http://arxiv.org/pdf/2508.12072v2)

**Authors**: Wei Jie Yeo, Ranjan Satapathy, Erik Cambria

**Abstract**: Despite extensive safety-tuning, large language models (LLMs) remain vulnerable to jailbreak attacks via adversarially crafted instructions, reflecting a persistent trade-off between safety and task performance. In this work, we propose Intent-FT, a simple and lightweight fine-tuning approach that explicitly trains LLMs to infer the underlying intent of an instruction before responding. By fine-tuning on a targeted set of adversarial instructions, Intent-FT enables LLMs to generalize intent deduction to unseen attacks, thereby substantially improving their robustness. We comprehensively evaluate both parametric and non-parametric attacks across open-source and proprietary models, considering harmfulness from attacks, utility, over-refusal, and impact against white-box threats. Empirically, Intent-FT consistently mitigates all evaluated attack categories, with no single attack exceeding a 50\% success rate -- whereas existing defenses remain only partially effective. Importantly, our method preserves the model's general capabilities and reduces excessive refusals on benign instructions containing superficially harmful keywords. Furthermore, models trained with Intent-FT accurately identify hidden harmful intent in adversarial attacks, and these learned intentions can be effectively transferred to enhance vanilla model defenses. We publicly release our code at https://github.com/wj210/Intent_Jailbreak.

摘要: 尽管进行了广泛的安全调整，大型语言模型（LLM）仍然容易受到通过敌对设计的指令的越狱攻击，这反映了安全性和任务性能之间的持续权衡。在这项工作中，我们提出了Intent-FT，这是一种简单且轻量级的微调方法，它在响应之前显式训练LLM推断指令的潜在意图。通过对目标对抗指令集进行微调，Intent-FT使LLM能够将意图演绎推广到不可见的攻击，从而大幅提高其稳健性。我们全面评估开源和专有模型中的参数和非参数攻击，考虑攻击的危害性、效用、过度拒绝以及对白盒威胁的影响。从经验上看，Intent-FT始终如一地减轻了所有评估的攻击类别，没有一次攻击的成功率超过50%，而现有的防御措施仅保持部分有效。重要的是，我们的方法保留了模型的一般功能，并减少了对包含表面有害关键词的良性指令的过度拒绝。此外，使用Intent-FT训练的模型可以准确识别对抗性攻击中隐藏的有害意图，并且可以有效地转移这些习得的意图以增强普通模型防御。我们在https://github.com/wj210/Intent_Jailbreak上公开发布我们的代码。



## **45. SALMAN: Stability Analysis of Language Models Through the Maps Between Graph-based Manifolds**

SALMAN：基于图流形间映射的语言模型稳定性分析 cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.18306v1) [paper-pdf](http://arxiv.org/pdf/2508.18306v1)

**Authors**: Wuxinlin Cheng, Yupeng Cao, Jinwen Wu, Koduvayur Subbalakshmi, Tian Han, Zhuo Feng

**Abstract**: Recent strides in pretrained transformer-based language models have propelled state-of-the-art performance in numerous NLP tasks. Yet, as these models grow in size and deployment, their robustness under input perturbations becomes an increasingly urgent question. Existing robustness methods often diverge between small-parameter and large-scale models (LLMs), and they typically rely on labor-intensive, sample-specific adversarial designs. In this paper, we propose a unified, local (sample-level) robustness framework (SALMAN) that evaluates model stability without modifying internal parameters or resorting to complex perturbation heuristics. Central to our approach is a novel Distance Mapping Distortion (DMD) measure, which ranks each sample's susceptibility by comparing input-to-output distance mappings in a near-linear complexity manner. By demonstrating significant gains in attack efficiency and robust training, we position our framework as a practical, model-agnostic tool for advancing the reliability of transformer-based NLP systems.

摘要: 预训练的基于转换器的语言模型最近取得了长足的进步，推动了众多NLP任务的最先进性能。然而，随着这些模型规模和部署的增长，它们在输入扰动下的稳健性成为一个日益紧迫的问题。现有的稳健性方法通常在小参数和大规模模型（LLM）之间存在分歧，并且它们通常依赖于劳动密集型、特定于样本的对抗性设计。在本文中，我们提出了一个统一的局部（样本级）鲁棒性框架（SALMAN），该框架无需修改内部参数或诉诸复杂的扰动试探法即可评估模型稳定性。我们方法的核心是一种新型的距离映射失真（DMZ）测量，该测量通过以近线性复杂性方式比较输入到输出距离映射来对每个样本的敏感性进行排名。通过展示攻击效率和稳健训练方面的显着提高，我们将我们的框架定位为一种实用的、模型不可知的工具，用于提高基于变压器的NLP系统的可靠性。



## **46. ImF: Implicit Fingerprint for Large Language Models**

ImF：大型语言模型的隐式指纹 cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2503.21805v3) [paper-pdf](http://arxiv.org/pdf/2503.21805v3)

**Authors**: Jiaxuan Wu, Wanli Peng, Hang Fu, Yiming Xue, Juan Wen

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing fingerprinting techniques typically embed identifiable patterns with weak semantic coherence, resulting in fingerprints that significantly differ from the natural question-answering (QA) behavior inherent to LLMs. This discrepancy undermines the stealthiness of the embedded fingerprints and makes them vulnerable to adversarial attacks. In this paper, we first demonstrate the critical vulnerability of existing fingerprint embedding methods by introducing a novel adversarial attack named Generation Revision Intervention (GRI) attack. GRI attack exploits the semantic fragility of current fingerprinting methods, effectively erasing fingerprints by disrupting their weakly correlated semantic structures. Our empirical evaluation highlights that traditional fingerprinting approaches are significantly compromised by the GRI attack, revealing severe limitations in their robustness under realistic adversarial conditions. To advance the state-of-the-art in model fingerprinting, we propose a novel model fingerprint paradigm called Implicit Fingerprints (ImF). ImF leverages steganography techniques to subtly embed ownership information within natural texts, subsequently using Chain-of-Thought (CoT) prompting to construct semantically coherent and contextually natural QA pairs. This design ensures that fingerprints seamlessly integrate with the standard model behavior, remaining indistinguishable from regular outputs and substantially reducing the risk of accidental triggering and targeted removal. We conduct a comprehensive evaluation of ImF on 15 diverse LLMs, spanning different architectures and varying scales.

摘要: 训练大型语言模型（LLM）是资源密集型且昂贵的，因此保护LLM的知识产权（IP）至关重要。最近，将指纹嵌入LLM已成为建立模型所有权的流行方法。然而，现有的指纹识别技术通常嵌入具有弱语义一致性的可识别模式，导致指纹与LLM固有的自然问答（QA）行为显着不同。这种差异削弱了嵌入指纹的隐蔽性，并使它们容易受到对抗攻击。在本文中，我们首先通过引入一种名为世代修订干预（GRI）攻击的新型对抗攻击来证明现有指纹嵌入方法的关键漏洞。GRI攻击利用了当前指纹识别方法的语义脆弱性，通过破坏指纹弱相关的语义结构来有效地擦除指纹。我们的经验评估强调，传统的指纹识别方法受到GRI攻击的严重损害，揭示了其在现实对抗条件下稳健性的严重局限性。为了推进模型指纹识别的最新水平，我们提出了一种新型模型指纹范式，称为隐式指纹（ImF）。ImF利用隐写技术将所有权信息巧妙地嵌入自然文本中，随后使用思想链（CoT）提示构建语义连贯且上下文自然的QA对。这种设计确保指纹与标准模型行为无缝集成，与常规输出保持无区别，并大幅降低意外触发和有针对性删除的风险。我们对15个不同的LLM进行了ImF的全面评估，涵盖不同的架构和不同的规模。



## **47. HAMSA: Hijacking Aligned Compact Models via Stealthy Automation**

HAMSA：通过隐形自动化劫持对齐的紧凑型车型 cs.CL

9 pages, 1 figure; article under review

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16484v1) [paper-pdf](http://arxiv.org/pdf/2508.16484v1)

**Authors**: Alexey Krylov, Iskander Vagizov, Dmitrii Korzh, Maryam Douiba, Azidine Guezzaz, Vladimir Kokh, Sergey D. Erokhin, Elena V. Tutubalina, Oleg Y. Rogov

**Abstract**: Large Language Models (LLMs), especially their compact efficiency-oriented variants, remain susceptible to jailbreak attacks that can elicit harmful outputs despite extensive alignment efforts. Existing adversarial prompt generation techniques often rely on manual engineering or rudimentary obfuscation, producing low-quality or incoherent text that is easily flagged by perplexity-based filters. We present an automated red-teaming framework that evolves semantically meaningful and stealthy jailbreak prompts for aligned compact LLMs. The approach employs a multi-stage evolutionary search, where candidate prompts are iteratively refined using a population-based strategy augmented with temperature-controlled variability to balance exploration and coherence preservation. This enables the systematic discovery of prompts capable of bypassing alignment safeguards while maintaining natural language fluency. We evaluate our method on benchmarks in English (In-The-Wild Jailbreak Prompts on LLMs), and a newly curated Arabic one derived from In-The-Wild Jailbreak Prompts on LLMs and annotated by native Arabic linguists, enabling multilingual assessment.

摘要: 大型语言模型（LLM），尤其是其紧凑的、以效率为导向的变体，仍然容易受到越狱攻击，尽管进行了广泛的对齐工作，这些攻击可能会引发有害的输出。现有的对抗性提示生成技术通常依赖于手动工程或基本混淆，从而产生低质量或不连贯的文本，这些文本很容易被基于困惑的过滤器标记。我们提出了一个自动化的红色团队框架，该框架进化出具有语义意义且隐蔽的越狱提示，以实现对齐的紧凑型LLM。该方法采用多阶段进化搜索，其中使用基于种群的策略迭代细化候选提示，并增强温度控制的变异性，以平衡探索和一致性保留。这使得系统性地发现能够绕过对齐保障措施，同时保持自然语言流利性的提示。我们用英语基准评估我们的方法（LLM上的In-The-Wild Jailbreak Pretts），以及一种新策划的阿拉伯语基准评估方法，该方法源自LLM上的In-The-Wild Jailbreak Pretts，并由母语阿拉伯语语言学家注释，从而实现多语言评估。



## **48. MCP-Guard: A Defense Framework for Model Context Protocol Integrity in Large Language Model Applications**

MCP-Guard：大型语言模型应用中模型上下文协议完整性的防御框架 cs.CR

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.10991v2) [paper-pdf](http://arxiv.org/pdf/2508.10991v2)

**Authors**: Wenpeng Xing, Zhonghao Qi, Yupeng Qin, Yilin Li, Caini Chang, Jiahui Yu, Changting Lin, Zhenzhen Xie, Meng Han

**Abstract**: The integration of Large Language Models (LLMs) with external tools via protocols such as the Model Context Protocol (MCP) introduces critical security vulnerabilities, including prompt injection, data exfiltration, and other threats. To counter these challenges, we propose MCP-Guard, a robust, layered defense architecture designed for LLM--tool interactions. MCP-Guard employs a three-stage detection pipeline that balances efficiency with accuracy: it progresses from lightweight static scanning for overt threats and a deep neural detector for semantic attacks, to our fine-tuned E5-based model achieves (96.01) accuracy in identifying adversarial prompts. Finally, a lightweight LLM arbitrator synthesizes these signals to deliver the final decision while minimizing false positives. To facilitate rigorous training and evaluation, we also introduce MCP-AttackBench, a comprehensive benchmark of over 70,000 samples. Sourced from public datasets and augmented by GPT-4, MCP-AttackBench simulates diverse, real-world attack vectors in the MCP format, providing a foundation for future research into securing LLM-tool ecosystems.

摘要: 通过模型上下文协议（HCP）等协议将大型语言模型（LLM）与外部工具集成会引入严重的安全漏洞，包括提示注入、数据溢出和其他威胁。为了应对这些挑战，我们提出了MCP-Guard，这是一种专为LLM工具交互而设计的稳健、分层的防御架构。MCP-Guard采用三阶段检测管道，平衡效率与准确性：它从针对明显威胁的轻量级静态扫描和针对语义攻击的深度神经检测器，发展到我们微调的基于E5的模型，在识别对抗性提示方面实现了（96.01）的准确性。最后，轻量级LLM仲裁器合成这些信号以做出最终决策，同时最大限度地减少误报。为了促进严格的培训和评估，我们还引入了MCP-AttackBench，这是一个包含超过70，000个样本的综合基准。MCP-AttackBench源自公共数据集，并通过GPT-4进行增强，以HCP格式模拟不同的现实世界攻击载体，为未来研究保护LLM工具生态系统提供基础。



## **49. Retrieval-Augmented Defense: Adaptive and Controllable Jailbreak Prevention for Large Language Models**

检索增强防御：大型语言模型的自适应且可控越狱预防 cs.CR

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16406v1) [paper-pdf](http://arxiv.org/pdf/2508.16406v1)

**Authors**: Guangyu Yang, Jinghong Chen, Jingbiao Mei, Weizhe Lin, Bill Byrne

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreak attacks, which attempt to elicit harmful responses from LLMs. The evolving nature and diversity of these attacks pose many challenges for defense systems, including (1) adaptation to counter emerging attack strategies without costly retraining, and (2) control of the trade-off between safety and utility. To address these challenges, we propose Retrieval-Augmented Defense (RAD), a novel framework for jailbreak detection that incorporates a database of known attack examples into Retrieval-Augmented Generation, which is used to infer the underlying, malicious user query and jailbreak strategy used to attack the system. RAD enables training-free updates for newly discovered jailbreak strategies and provides a mechanism to balance safety and utility. Experiments on StrongREJECT show that RAD substantially reduces the effectiveness of strong jailbreak attacks such as PAP and PAIR while maintaining low rejection rates for benign queries. We propose a novel evaluation scheme and show that RAD achieves a robust safety-utility trade-off across a range of operating points in a controllable manner.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，这些攻击试图引发LLM的有害反应。这些攻击不断变化的性质和多样性给防御系统带来了许多挑战，包括（1）在无需昂贵的再培训的情况下适应应对新兴攻击策略，以及（2）控制安全性和实用性之间的权衡。为了应对这些挑战，我们提出了检索增强防御（RAD），这是一种新颖的越狱检测框架，将已知攻击示例的数据库整合到检索增强生成中，用于推断底层的恶意用户查询和用于攻击系统的越狱策略。RAD为新发现的越狱策略提供免训练更新，并提供平衡安全性和实用性的机制。StrongRESEARCH上的实验表明，RAD大大降低了PAP和PAIR等强越狱攻击的有效性，同时保持良性查询的低拒绝率。我们提出了一种新颖的评估方案，并表明RAD以可控的方式在一系列操作点上实现了稳健的安全-效用权衡。



## **50. Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs**

混乱是最后的障碍：重新思考越狱评估并调查LLM的真正滥用威胁 cs.CR

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16347v1) [paper-pdf](http://arxiv.org/pdf/2508.16347v1)

**Authors**: Yu Yan, Sheng Sun, Zhe Wang, Yijun Lin, Zenghao Duan, zhifei zheng, Min Liu, Zhiyi yin, Jianping Zhang

**Abstract**: With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential.

摘要: 随着大型语言模型（LLM）的发展，许多努力揭示了它们对越狱攻击的脆弱性。尽管这些研究推动了LLM安全调整的进展，但目前尚不清楚LLM是否已经内化了真实的知识来应对现实世界的犯罪，或者只是被迫模拟有毒的语言模式。这种模糊性引发了人们的担忧，即越狱成功通常归因于越狱LLM和法官LLM之间的幻觉循环。通过脱钩越狱技术的使用，我们构建知识密集型问答，以调查LLM在危险知识拥有、有害任务规划效用和有害判断稳健性方面的滥用威胁。实验揭示了LLM的越狱成功率和有害知识拥有之间的不匹配，而现有的LLM作为法官框架往往会将有害判断锚定在有毒语言模式上。我们的研究揭示了现有LLM安全评估与现实世界潜在威胁之间的差距。



