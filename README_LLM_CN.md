# Latest Large Language Model Attack Papers
**update at 2025-07-05 11:48:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection**

视觉上下文攻击：利用图像驱动上下文注入越狱MLLM cs.CV

16 pages

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02844v1) [paper-pdf](http://arxiv.org/pdf/2507.02844v1)

**Authors**: Ziqi Miao, Yi Ding, Lijun Li, Jing Shao

**Abstract**: With the emergence of strong visual-language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: visual-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct visual-focused strategies, dynamically generating auxiliary images when necessary to construct a visual-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which performs a toxicity score of 2.48 and an ASR of 22.2%. The code is available at https://github.com/Dtc7w3PQ/Visco-Attack.

摘要: 随着强大的视觉语言能力的出现，多模式大型语言模型（MLLM）在现实世界应用中展示了巨大的潜力。然而，视觉模式所表现出的安全漏洞对在开放世界环境中部署此类模型构成了重大挑战。最近的研究通过将有害的文本语义直接编码到视觉输入中，成功地诱导了目标MLLM的有害反应。然而，在这些方法中，视觉形态主要充当不安全行为的触发器，通常表现出语义模糊性并且在现实场景中缺乏基础。在这项工作中，我们定义了一种新颖的环境：以视觉为中心的越狱，其中视觉信息是构建完整而现实的越狱背景的必要组成部分。在此设置的基础上，我们提出了VisCo（视觉上下文）攻击。VisCo使用四种不同的以视觉为中心的策略构建上下文对话，在必要时动态生成辅助图像以构建以视觉为中心的越狱场景。为了最大限度地提高攻击效果，它结合了自动毒性混淆和语义细化，以产生最终的攻击提示，从而可靠地触发目标黑匣子MLLM的有害响应。具体而言，VisCo在MM-SafetyBench上针对GPT-4 o实现了4.78的毒性评分和85%的攻击成功率（ASR），显著优于基线，其毒性评分为2.48，ASR为22.2%。该代码可在https://github.com/Dtc7w3PQ/Visco-Attack上获取。



## **2. Is Reasoning All You Need? Probing Bias in the Age of Reasoning Language Models**

你需要的就是推理吗？探索推理语言模型时代的偏见 cs.CL

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02799v1) [paper-pdf](http://arxiv.org/pdf/2507.02799v1)

**Authors**: Riccardo Cantini, Nicola Gabriele, Alessio Orsino, Domenico Talia

**Abstract**: Reasoning Language Models (RLMs) have gained traction for their ability to perform complex, multi-step reasoning tasks through mechanisms such as Chain-of-Thought (CoT) prompting or fine-tuned reasoning traces. While these capabilities promise improved reliability, their impact on robustness to social biases remains unclear. In this work, we leverage the CLEAR-Bias benchmark, originally designed for Large Language Models (LLMs), to investigate the adversarial robustness of RLMs to bias elicitation. We systematically evaluate state-of-the-art RLMs across diverse sociocultural dimensions, using an LLM-as-a-judge approach for automated safety scoring and leveraging jailbreak techniques to assess the strength of built-in safety mechanisms. Our evaluation addresses three key questions: (i) how the introduction of reasoning capabilities affects model fairness and robustness; (ii) whether models fine-tuned for reasoning exhibit greater safety than those relying on CoT prompting at inference time; and (iii) how the success rate of jailbreak attacks targeting bias elicitation varies with the reasoning mechanisms employed. Our findings reveal a nuanced relationship between reasoning capabilities and bias safety. Surprisingly, models with explicit reasoning, whether via CoT prompting or fine-tuned reasoning traces, are generally more vulnerable to bias elicitation than base models without such mechanisms, suggesting reasoning may unintentionally open new pathways for stereotype reinforcement. Reasoning-enabled models appear somewhat safer than those relying on CoT prompting, which are particularly prone to contextual reframing attacks through storytelling prompts, fictional personas, or reward-shaped instructions. These results challenge the assumption that reasoning inherently improves robustness and underscore the need for more bias-aware approaches to reasoning design.

摘要: 推理语言模型（RLM）通过诸如思想链（CoT）提示或微调推理轨迹等机制来执行复杂的多步推理任务的能力已经获得了关注。虽然这些功能有望提高可靠性，但它们对社会偏见鲁棒性的影响仍不清楚。在这项工作中，我们利用CLEAR-Bias基准，最初是为大型语言模型（LLM）设计的，来研究RLM对偏见启发的对抗鲁棒性。我们在不同的社会文化维度上系统地评估最先进的RLM，使用LLM作为自动安全评分的评判方法，并利用越狱技术来评估内置安全机制的强度。我们的评估解决了三个关键问题：（i）推理能力的引入如何影响模型的公平性和稳健性;（ii）为推理进行微调的模型是否比在推理时依赖CoT提示的模型表现出更大的安全性;（iii）针对偏见引发的越狱攻击的成功率如何随着所采用的推理机制而变化。我们的研究结果揭示了推理能力和偏见安全性之间的微妙关系。令人惊讶的是，具有显式推理的模型，无论是通过CoT提示还是微调推理痕迹，通常比没有此类机制的基本模型更容易受到偏见引发，这表明推理可能会无意中为刻板印象强化开辟新的途径。支持推理的模型似乎比依赖CoT提示的模型更安全，后者特别容易受到通过讲故事提示、虚构人物角色或奖励形状指令的上下文重组攻击。这些结果挑战了推理本质上可以提高稳健性的假设，并强调了对推理设计的更多偏差感知方法的需求。



## **3. StructTransform: A Scalable Attack Surface for Safety-Aligned Large Language Models**

StructChange：安全一致的大型语言模型的可扩展攻击表面 cs.LG

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2502.11853v2) [paper-pdf](http://arxiv.org/pdf/2502.11853v2)

**Authors**: Shehel Yoosuf, Temoor Ali, Ahmed Lekssays, Mashael AlSabah, Issa Khalil

**Abstract**: In this work, we present a series of structure transformation attacks on LLM alignment, where we encode natural language intent using diverse syntax spaces, ranging from simple structure formats and basic query languages (e.g., SQL) to new novel spaces and syntaxes created entirely by LLMs. Our extensive evaluation shows that our simplest attacks can achieve close to a 90% success rate, even on strict LLMs (such as Claude 3.5 Sonnet) using SOTA alignment mechanisms. We improve the attack performance further by using an adaptive scheme that combines structure transformations along with existing content transformations, resulting in over 96% ASR with 0% refusals.   To generalize our attacks, we explore numerous structure formats, including syntaxes purely generated by LLMs. Our results indicate that such novel syntaxes are easy to generate and result in a high ASR, suggesting that defending against our attacks is not a straightforward process. Finally, we develop a benchmark and evaluate existing safety-alignment defenses against it, showing that most of them fail with 100% ASR. Our results show that existing safety alignment mostly relies on token-level patterns without recognizing harmful concepts, highlighting and motivating the need for serious research efforts in this direction. As a case study, we demonstrate how attackers can use our attack to easily generate a sample malware and a corpus of fraudulent SMS messages, which perform well in bypassing detection.

摘要: 在这项工作中，我们提出了一系列对LLM对齐的结构转换攻击，其中我们使用不同的语法空间对自然语言意图进行编码，从简单的结构格式到基本的查询语言（例如，SQL）到完全由LLM创建的新空间和语法。我们广泛的评估表明，我们最简单的攻击可以达到接近90%的成功率，即使是在严格的LLM（如Claude 3.5 Sonnet）上使用SOTA对齐机制。我们进一步提高攻击性能，通过使用自适应方案，结合结构转换与现有的内容转换，导致超过96%的ASR与0%的拒绝。   为了概括我们的攻击，我们探索了多种结构格式，包括纯粹由LLM生成的语法。我们的结果表明，这种新颖的语法很容易生成并导致高的ASB，这表明防御我们的攻击并不是一个简单的过程。最后，我们开发了一个基准并评估了现有的安全对齐防御，表明其中大多数都在100%的ASC下失败。我们的结果表明，现有的安全调整主要依赖于代币级模式，而没有识别出有害的概念，凸显并激励了在这一方向进行认真研究的必要性。作为案例研究，我们展示了攻击者如何使用我们的攻击来轻松生成恶意软件样本和欺诈性短信消息集，这些信息在绕过检测方面表现良好。



## **4. Evaluating Language Models For Threat Detection in IoT Security Logs**

评估语言模型用于物联网安全威胁检测 cs.CR

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02390v1) [paper-pdf](http://arxiv.org/pdf/2507.02390v1)

**Authors**: Jorge J. Tejero-Fernández, Alfonso Sánchez-Macián

**Abstract**: Log analysis is a relevant research field in cybersecurity as they can provide a source of information for the detection of threats to networks and systems. This paper presents a pipeline to use fine-tuned Large Language Models (LLMs) for anomaly detection and mitigation recommendation using IoT security logs. Utilizing classical machine learning classifiers as a baseline, three open-source LLMs are compared for binary and multiclass anomaly detection, with three strategies: zero-shot, few-shot prompting and fine-tuning using an IoT dataset. LLMs give better results on multi-class attack classification than the corresponding baseline models. By mapping detected threats to MITRE CAPEC, defining a set of IoT-specific mitigation actions, and fine-tuning the models with those actions, the models are able to provide a combined detection and recommendation guidance.

摘要: 日志分析是网络安全中的一个相关研究领域，因为它们可以为检测网络和系统的威胁提供信息来源。本文提出了一种管道，使用微调的大型语言模型（LLM），使用物联网安全日志进行异常检测和缓解建议。利用经典的机器学习分类器作为基线，比较了三种开源LLM的二进制和多类异常检测，采用三种策略：零次、少量提示和使用物联网数据集进行微调。LLM在多类攻击分类上比相应的基线模型给出了更好的结果。通过将检测到的威胁映射到MITRE CAPEC，定义一组特定于物联网的缓解措施，并使用这些措施微调模型，这些模型能够提供组合的检测和建议指导。



## **5. SecAlign: Defending Against Prompt Injection with Preference Optimization**

SecAlign：通过偏好优化抵御提示注入 cs.CR

ACM CCS 2025. Key words: prompt injection defense, LLM security,  LLM-integrated applications

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2410.05451v3) [paper-pdf](http://arxiv.org/pdf/2410.05451v3)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the Internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be injected into external data sources to override the system's intended instruction and instead execute a malicious instruction. To mitigate this vulnerability, we propose a new defense called SecAlign based on the technique of preference optimization. Our defense first constructs a preference dataset with prompt-injected inputs, secure outputs (ones that respond to the legitimate instruction), and insecure outputs (ones that respond to the injection). We then perform preference optimization on this dataset to teach the LLM to prefer the secure output over the insecure one. This provides the first known method that reduces the success rates of various prompt injections to <10%, even against attacks much more sophisticated than ones seen during training. This indicates our defense generalizes well against unknown and yet-to-come attacks. Also, SecAlign models are still practical with similar utility to the one before defensive training in our evaluations. Our code is at https://github.com/facebookresearch/SecAlign

摘要: 大型语言模型（LLM）在现代软件系统中变得越来越普遍，在用户和互联网之间进行接口，以协助执行需要高级语言理解的任务。为了完成这些任务，LLM通常使用外部数据源，例如用户文档、Web检索、API调用的结果等。这为攻击者通过提示注入操纵LLM开辟了新的途径。对抗性提示可以被注入到外部数据源中，以覆盖系统的预期指令，转而执行恶意指令。为了缓解此漏洞，我们基于偏好优化技术提出了一种名为SecAlign的新防御。我们的防御首先构建一个具有预算注入的输入、安全输出（响应合法指令的输出）和不安全输出（响应注入的输出）的偏好数据集。然后，我们对该数据集执行偏好优化，以教导LLM更喜欢安全的输出而不是不安全的输出。这提供了第一种已知的方法，可以将各种即时注射的成功率降低到<10%，即使是针对比训练期间看到的攻击复杂得多的攻击。这表明我们的防御对于未知和尚未到来的攻击具有很好的概括性。此外，SecAlign模型仍然实用，与我们评估中防御训练前的模型相似。我们的代码位于https://github.com/facebookresearch/SecAlign



## **6. MGC: A Compiler Framework Exploiting Compositional Blindness in Aligned LLMs for Malware Generation**

MCR：一个更简单的框架，利用对齐的LLM中的合成盲度来生成恶意软件 cs.CR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.02057v1) [paper-pdf](http://arxiv.org/pdf/2507.02057v1)

**Authors**: Lu Yan, Zhuo Zhang, Xiangzhe Xu, Shengwei An, Guangyu Shen, Zhou Xuan, Xuan Chen, Xiangyu Zhang

**Abstract**: Large language models (LLMs) have democratized software development, reducing the expertise barrier for programming complex applications. This accessibility extends to malicious software development, raising significant security concerns. While LLM providers have implemented alignment mechanisms to prevent direct generation of overtly malicious code, these safeguards predominantly evaluate individual prompts in isolation, overlooking a critical vulnerability: malicious operations can be systematically decomposed into benign-appearing sub-tasks. In this paper, we introduce the Malware Generation Compiler (MGC), a novel framework that leverages this vulnerability through modular decomposition and alignment-evasive generation. MGC employs a specialized Malware Description Intermediate Representation (MDIR) to bridge high-level malicious intents and benign-appearing code snippets. Extensive evaluation demonstrates that our attack reliably generates functional malware across diverse task specifications and categories, outperforming jailbreaking methods by +365.79% and underground services by +78.07% in correctness on three benchmark datasets. Case studies further show that MGC can reproduce and even enhance 16 real-world malware samples. This work provides critical insights for security researchers by exposing the risks of compositional attacks against aligned AI systems. Demonstrations are available at https://sites.google.com/view/malware-generation-compiler.

摘要: 大型语言模型（LLM）使软件开发民主化，减少了编程复杂应用程序的专业知识障碍。这种可访问性扩展到恶意软件开发，引发了严重的安全问题。虽然LLM提供商已经实施了对齐机制来防止直接生成明显的恶意代码，但这些保护措施主要孤立地评估单个提示，忽略了一个关键漏洞：恶意操作可以系统地分解为看似善意的子任务。在本文中，我们介绍了恶意软件生成漏洞（MRC），这是一个新颖的框架，通过模块化分解和漏洞规避生成来利用该漏洞。MGC采用专门的恶意软件描述中间表示（Malware Description Intermediate Representation，MPEG4）来桥接高级恶意意图和善意代码片段。广泛的评估表明，我们的攻击可以可靠地生成跨不同任务规范和类别的功能性恶意软件，在三个基准数据集上的正确性超过越狱方法+365.79%和地下服务+78.07%。案例研究进一步表明，MGC可以复制甚至增强16个真实世界的恶意软件样本。这项工作为安全研究人员提供了重要的见解，揭示了针对对齐AI系统的组合攻击的风险。演示可在https://sites.google.com/view/malware-generation-compiler上获取。



## **7. Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training**

没有偷看的调整：LLM后培训的可证明隐私和泛化边界 cs.LG

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01752v1) [paper-pdf](http://arxiv.org/pdf/2507.01752v1)

**Authors**: Ismail Labiad, Mathurin Videau, Matthieu Kowalski, Marc Schoenauer, Alessandro Leite, Julia Kempe, Olivier Teytaud

**Abstract**: Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, its reliance on large volumes of labeled data raises privacy and security concerns such as susceptibility to data poisoning attacks and the risk of overfitting. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. However, black box methods also pose significant challenges, including poor scalability to high-dimensional parameter spaces, as prevalent in large language models (LLMs), and high computational costs due to reliance on numerous model evaluations. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide strong theoretical bounds on generalization, differential privacy, susceptibility to data poisoning attacks, and robustness to extraction attacks. BBoxER operates on top of pre-trained LLMs, offering a lightweight and modular enhancement suitable for deployment in restricted or privacy-sensitive environments, in addition to non-vacuous generalization guarantees. In experiments with LLMs, we demonstrate empirically that Retrofitting methods are able to learn, showing how a few iterations of BBoxER improve performance and generalize well on a benchmark of reasoning datasets. This positions BBoxER as an attractive add-on on top of gradient-based optimization.

摘要: 基于对象的优化是深度学习的主力，通过反向传播提供高效且可扩展的训练。然而，它对大量标记数据的依赖引发了隐私和安全问题，例如容易受到数据中毒攻击和过度匹配的风险。相比之下，黑匣子优化方法将模型视为一个不透明的函数，仅依赖函数评估来指导优化，在数据访问受到限制、对抗风险较高或过度匹配令人担忧的场景中提供了一种有希望的替代方案。然而，黑匣子方法也带来了重大挑战，包括大型语言模型（LLM）中普遍存在的对多维参数空间的可扩展性较差，以及由于依赖大量模型评估而导致的高计算成本。本文介绍了BBoxER，这是一种用于LLM后训练的进化黑匣子方法，通过隐式压缩训练数据来引发信息瓶颈。利用信息流的可追溯性，我们在概括性、差异隐私、对数据中毒攻击的敏感性以及对提取攻击的鲁棒性方面提供了强大的理论界限。BBoxER在预先培训的LLM之上运行，除了非空洞的通用保证外，还提供适合在受限制或隐私敏感环境中部署的轻量级模块化增强。在LLM的实验中，我们经验地证明了Retrofit方法能够学习，展示了BBoxER的几次迭代如何提高性能并在推理数据集的基准上很好地概括。这使得BBoxER成为基于梯度的优化之上的一个有吸引力的附加组件。



## **8. Graph Representation-based Model Poisoning on Federated LLMs in CyberEdge Networks**

CyberEdge网络中联邦LLM上基于图表示的模型中毒 cs.CR

7 pages, 5 figures

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01694v1) [paper-pdf](http://arxiv.org/pdf/2507.01694v1)

**Authors**: Hanlin Cai, Haofan Dong, Houtianfu Wang, Kai Li, Ozgur B. Akan

**Abstract**: Federated large language models (FedLLMs) provide powerful generative capabilities in CyberEdge networks while protecting data privacy. However, FedLLMs remains highly vulnerable to model poisoning attacks. This article first reviews recent model poisoning techniques and existing defense mechanisms for FedLLMs, highlighting critical limitations, particularly under non-IID text distributions. In particular, current defenses primarily utilize distance-based outlier detection or norm constraints, operating under the assumption that adversarial updates significantly diverge from benign statistics. This assumption can fail when facing adaptive attackers targeting billionparameter LLMs. Next, this article investigates emerging Graph Representation-Based Model Poisoning (GRMP), a novel attack paradigm that leverages higher-order correlations among honest client gradients to synthesize malicious updates indistinguishable from legitimate model updates. GRMP can effectively evade advanced defenses, resulting in substantial accuracy loss and performance degradation. Moreover, this article outlines a research roadmap emphasizing the importance of graph-aware secure aggregation methods, FedLLMs-specific vulnerability metrics, and evaluation frameworks to strengthen the robustness of future federated language model deployments.

摘要: 联合大型语言模型（FedLLM）在CyberEdge网络中提供强大的生成能力，同时保护数据隐私。然而，FedLLM仍然极易受到模型中毒攻击。本文首先回顾了FedLLM最近的模型中毒技术和现有的防御机制，强调了关键的局限性，特别是在非IID文本分发下。特别是，当前的防御主要利用基于距离的离群值检测或规范约束，在对抗性更新与良性统计数据显着偏离的假设下运行。当面对针对十亿参数LLM的自适应攻击者时，这一假设可能会失败。接下来，本文研究了新兴的基于图表示的模型中毒（GRMP），这是一种新型攻击范式，它利用诚实客户端梯度之间的更高层相关性来合成与合法模型更新没有区别的恶意更新。GRMP可以有效规避高级防御，导致准确性大幅损失和性能下降。此外，本文还概述了一份研究路线图，强调图形感知的安全聚合方法、特定于FedLLM的漏洞指标和评估框架的重要性，以加强未来联邦语言模型部署的稳健性。



## **9. SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism**

SafeTLR：通过删除然后恢复机制在多模式LLM中进行令牌级越狱防御 cs.CR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01513v1) [paper-pdf](http://arxiv.org/pdf/2507.01513v1)

**Authors**: Beitao Chen, Xinyu Lyu, Lianli Gao, Jingkuan Song, Heng Tao Shen

**Abstract**: By incorporating visual inputs, Multimodal Large Language Models (MLLMs) extend LLMs to support visual reasoning. However, this integration also introduces new vulnerabilities, making MLLMs susceptible to multimodal jailbreak attacks and hindering their safe deployment.Existing defense methods, including Image-to-Text Translation, Safe Prompting, and Multimodal Safety Tuning, attempt to address this by aligning multimodal inputs with LLMs' built-in safeguards.Yet, they fall short in uncovering root causes of multimodal vulnerabilities, particularly how harmful multimodal tokens trigger jailbreak in MLLMs? Consequently, they remain vulnerable to text-driven multimodal jailbreaks, often exhibiting overdefensive behaviors and imposing heavy training overhead.To bridge this gap, we present an comprehensive analysis of where, how and which harmful multimodal tokens bypass safeguards in MLLMs. Surprisingly, we find that less than 1% tokens in early-middle layers are responsible for inducing unsafe behaviors, highlighting the potential of precisely removing a small subset of harmful tokens, without requiring safety tuning, can still effectively improve safety against jailbreaks. Motivated by this, we propose Safe Prune-then-Restore (SafePTR), an training-free defense framework that selectively prunes harmful tokens at vulnerable layers while restoring benign features at subsequent layers.Without incurring additional computational overhead, SafePTR significantly enhances the safety of MLLMs while preserving efficiency. Extensive evaluations across three MLLMs and five benchmarks demonstrate SafePTR's state-of-the-art performance in mitigating jailbreak risks without compromising utility.

摘要: 通过结合视觉输入，多模式大型语言模型（MLLM）扩展了LLM以支持视觉推理。然而，这种集成也引入了新的漏洞，使MLLM容易受到多模式越狱攻击并阻碍其安全部署。现有的防御方法，包括图像到文本翻译、安全预算处理和多模式安全调优，试图通过将多模式输入与LLM的内置保护措施相一致来解决这个问题。然而，它们未能发现多模式漏洞的根本原因，特别是有害的多模式代币如何触发MLLM越狱？因此，他们仍然容易受到文本驱动的多模式越狱的影响，通常表现出过度防御行为并施加沉重的培训费用。为了弥合这一差距，我们对MLLM中的哪些有害多模式代币在哪里、如何以及哪些方式绕过保障措施进行了全面分析。令人惊讶的是，我们发现，在早期-中间层中，只有不到1%的令牌会导致不安全的行为，这突出了精确删除一小部分有害令牌的潜力，而不需要进行安全调整，仍然可以有效地提高安全性。基于此，我们提出了安全修剪然后恢复（SafePTR），这是一个无需训练的防御框架，它可以选择性地修剪脆弱层的有害令牌，同时恢复后续层的良性特征。在不产生额外计算开销的情况下，SafePTR显著增强了MLLM的安全性，同时保持了效率。三个MLLM和五个基准测试的广泛评估证明了SafePTR在减轻越狱风险而不影响实用性方面的最先进性能。



## **10. Don't Say No: Jailbreaking LLM by Suppressing Refusal**

不要说不：通过压制拒绝来越狱法学硕士 cs.CL

Accepted by ACL 2025 Findings

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2404.16369v3) [paper-pdf](http://arxiv.org/pdf/2404.16369v3)

**Authors**: Yukai Zhou, Jian Lou, Zhijie Huang, Zhan Qin, Yibei Yang, Wenjie Wang

**Abstract**: Ensuring the safety alignment of Large Language Models (LLMs) is critical for generating responses consistent with human values. However, LLMs remain vulnerable to jailbreaking attacks, where carefully crafted prompts manipulate them into producing toxic content. One category of such attacks reformulates the task as an optimization problem, aiming to elicit affirmative responses from the LLM. However, these methods heavily rely on predefined objectionable behaviors, limiting their effectiveness and adaptability to diverse harmful queries. In this study, we first identify why the vanilla target loss is suboptimal and then propose enhancements to the loss objective. We introduce DSN (Don't Say No) attack, which combines a cosine decay schedule method with refusal suppression to achieve higher success rates. Extensive experiments demonstrate that DSN outperforms baseline attacks and achieves state-of-the-art attack success rates (ASR). DSN also shows strong universality and transferability to unseen datasets and black-box models.

摘要: 确保大型语言模型（LLM）的安全一致对于生成与人类价值观一致的响应至关重要。然而，LLM仍然容易受到越狱攻击，精心设计的提示操纵它们产生有毒内容。一类此类攻击将任务重新定义为优化问题，旨在引起LLM的肯定响应。然而，这些方法严重依赖于预定义的不良行为，限制了它们对各种有害查询的有效性和适应性。在这项研究中，我们首先确定为什么香草目标损失不是最优的，然后提出对损失目标的增强措施。我们引入了SEN（Don ' t Say No）攻击，该攻击将cos衰变调度方法与拒绝抑制相结合，以实现更高的成功率。大量实验表明，SEN的性能优于基线攻击，并实现了最先进的攻击成功率（ASB）。SEN还表现出强大的通用性和对未见数据集和黑匣子模型的可移植性。



## **11. ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks**

ICLShield：探索和缓解上下文学习后门攻击 cs.LG

ICML 2025

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01321v1) [paper-pdf](http://arxiv.org/pdf/2507.01321v1)

**Authors**: Zhiyao Ren, Siyuan Liang, Aishan Liu, Dacheng Tao

**Abstract**: In-context learning (ICL) has demonstrated remarkable success in large language models (LLMs) due to its adaptability and parameter-free nature. However, it also introduces a critical vulnerability to backdoor attacks, where adversaries can manipulate LLM behaviors by simply poisoning a few ICL demonstrations. In this paper, we propose, for the first time, the dual-learning hypothesis, which posits that LLMs simultaneously learn both the task-relevant latent concepts and backdoor latent concepts within poisoned demonstrations, jointly influencing the probability of model outputs. Through theoretical analysis, we derive an upper bound for ICL backdoor effects, revealing that the vulnerability is dominated by the concept preference ratio between the task and the backdoor. Motivated by these findings, we propose ICLShield, a defense mechanism that dynamically adjusts the concept preference ratio. Our method encourages LLMs to select clean demonstrations during the ICL phase by leveraging confidence and similarity scores, effectively mitigating susceptibility to backdoor attacks. Extensive experiments across multiple LLMs and tasks demonstrate that our method achieves state-of-the-art defense effectiveness, significantly outperforming existing approaches (+26.02% on average). Furthermore, our method exhibits exceptional adaptability and defensive performance even for closed-source models (e.g., GPT-4).

摘要: 上下文学习（ICL）因其适应性和无参数性质而在大型语言模型（LLM）中取得了显着的成功。然而，它也引入了后门攻击的关键漏洞，对手可以通过简单地毒害一些ICL演示来操纵LLM行为。在本文中，我们首次提出了双重学习假设，该假设LLM同时学习与任务相关的潜在概念和中毒演示中的后门潜在概念，共同影响模型输出的可能性。通过理论分析，我们推导出ICL后门效应的上界，揭示了漏洞由任务与后门之间的概念偏好比决定。受这些发现的启发，我们提出了ICLShield，这是一种动态调整概念偏好比的防御机制。我们的方法鼓励LLM通过利用置信度和相似性分数在ICL阶段选择干净的演示，从而有效地降低对后门攻击的敏感性。跨多个LLM和任务的广泛实验表明，我们的方法实现了最先进的防御有效性，显着优于现有方法（平均+26.02%）。此外，即使对于闭源模型（例如，GPT-4）。



## **12. GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs**

GenBFA：对LLM进行位翻转攻击的进化优化方法 cs.CR

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2411.13757v4) [paper-pdf](http://arxiv.org/pdf/2411.13757v4)

**Authors**: Sanjay Das, Swastik Bhattacharya, Souvik Kundu, Shamik Kundu, Anand Menon, Arnab Raha, Kanad Basu

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.

摘要: 大型语言模型（LLM）彻底改变了自然语言处理（NLP），在文本生成和摘要等任务方面表现出色。然而，它们在任务关键型应用程序中的越来越多的采用引发了人们对基于硬件的威胁的担忧，特别是位翻转攻击（BFA）。BFA由Rowhammer等故障注入方法启用，目标是内存中的模型参数，从而损害完整性和性能。在LLM的巨大参数空间中识别BFA的关键参数构成了重大挑战。虽然之前的研究表明，与传统的深度神经网络相比，基于变换器的架构本质上对BFA更稳健，但我们挑战了这一假设。我们首次证明，在具有数十亿个参数的LLM中，只要三个位翻转就可能导致灾难性的性能下降。由于难以在巨大的参数空间中有效识别关键参数，目前的BFA技术不足以利用该漏洞。为了解决这个问题，我们提出了AttentionBreaker，这是一个为LLM量身定制的新型框架，可以有效地穿越参数空间以识别关键参数。此外，我们还引入了GenBFA，这是一种进化优化策略，旨在进一步细化搜索，隔离最关键的部分以进行高效且有效的攻击。实证结果揭示了LLM对AttentionBreaker的严重脆弱性。例如，LLaMA 3 - 8B-Direcct 8位量化（W8）模型中仅进行三次位翻转（总参数的4.129 x 10 '-9%）就会导致性能完全崩溃：MMLU任务的准确性从67.3%下降到0%，维基文本困惑度从12.6飙升到4.72 x 105。这些发现强调了AttentionBreaker在发现和利用LLM架构中关键漏洞方面的有效性。



## **13. Defensive Adversarial CAPTCHA: A Semantics-Driven Framework for Natural Adversarial Example Generation**

防御性对抗验证码：用于自然对抗示例生成的语义驱动框架 cs.CV

13 pages, 6 figures

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2506.10685v3) [paper-pdf](http://arxiv.org/pdf/2506.10685v3)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Cong Wu, Tao Li, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: Traditional CAPTCHA (Completely Automated Public Turing Test to Tell Computers and Humans Apart) schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on the original image characteristics, resulting in distortions that hinder human interpretation and limit their applicability in scenarios where no initial input images are available. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (DAC), a novel framework that generates high-fidelity adversarial examples guided by attacker-specified semantics information. Leveraging a Large Language Model (LLM), DAC enhances CAPTCHA diversity and enriches the semantic information. To address various application scenarios, we examine the white-box targeted attack scenario and the black box untargeted attack scenario. For target attacks, we introduce two latent noise variables that are alternately guided in the diffusion step to achieve robust inversion. The synergy between gradient guidance and latent variable optimization achieved in this way ensures that the generated adversarial examples not only accurately align with the target conditions but also achieve optimal performance in terms of distributional consistency and attack effectiveness. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-DAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show that the defensive adversarial CAPTCHA generated by BP-DAC is able to defend against most of the unknown models, and the generated CAPTCHA is indistinguishable to both humans and DNNs.

摘要: 传统的CAPTCHA（完全自动化公共图灵测试来区分计算机和人类）计划越来越容易受到深度神经网络（DNN）支持的自动化攻击。现有的对抗攻击方法通常依赖于原始图像特征，从而导致失真，阻碍人类解释并限制其在没有初始输入图像可用的场景中的适用性。为了解决这些挑战，我们提出了无源对抗性验证码（ADC），这是一种新颖的框架，可以在攻击者指定的语义信息的指导下生成高保真对抗性示例。利用大型语言模型（LLM），DEC增强了CAPTCHA的多样性并丰富了语义信息。为了应对各种应用场景，我们研究了白盒定向攻击场景和黑匣子非定向攻击场景。对于目标攻击，我们引入了两个潜在噪音变量，它们在扩散步骤中交替引导，以实现鲁棒的反转。通过这种方式实现的梯度引导和潜在变量优化之间的协同作用，确保生成的对抗示例不仅与目标条件准确对齐，而且在分布一致性和攻击有效性方面实现最佳性能。在无目标攻击中，特别是对于黑匣子场景，我们引入了双路径无源对抗性CAPTCHA（BP-ADC），这是一种两步优化策略，采用多峰梯度和双路径优化来实现高效的误分类。实验表明，BP-ADC生成的防御性对抗CAPTCHA能够防御大多数未知模型，并且生成的CAPTCHA对于人类和DNN来说都无法区分。



## **14. SafeMobile: Chain-level Jailbreak Detection and Automated Evaluation for Multimodal Mobile Agents**

SafeMobile：多模式移动代理的连锁级越狱检测和自动评估 cs.AI

12 pages

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00841v1) [paper-pdf](http://arxiv.org/pdf/2507.00841v1)

**Authors**: Siyuan Liang, Tianmeng Fang, Zhe Liu, Aishan Liu, Yan Xiao, Jinyuan He, Ee-Chien Chang, Xiaochun Cao

**Abstract**: With the wide application of multimodal foundation models in intelligent agent systems, scenarios such as mobile device control, intelligent assistant interaction, and multimodal task execution are gradually relying on such large model-driven agents. However, the related systems are also increasingly exposed to potential jailbreak risks. Attackers may induce the agents to bypass the original behavioral constraints through specific inputs, and then trigger certain risky and sensitive operations, such as modifying settings, executing unauthorized commands, or impersonating user identities, which brings new challenges to system security. Existing security measures for intelligent agents still have limitations when facing complex interactions, especially in detecting potentially risky behaviors across multiple rounds of conversations or sequences of tasks. In addition, an efficient and consistent automated methodology to assist in assessing and determining the impact of such risks is currently lacking. This work explores the security issues surrounding mobile multimodal agents, attempts to construct a risk discrimination mechanism by incorporating behavioral sequence information, and designs an automated assisted assessment scheme based on a large language model. Through preliminary validation in several representative high-risk tasks, the results show that the method can improve the recognition of risky behaviors to some extent and assist in reducing the probability of agents being jailbroken. We hope that this study can provide some valuable references for the security risk modeling and protection of multimodal intelligent agent systems.

摘要: 随着多模式基础模型在智能代理系统中的广泛应用，移动终端控制、智能助理交互、多模式任务执行等场景逐渐依赖于此类大型模型驱动的代理。然而，相关系统也越来越多地面临潜在的越狱风险。攻击者可能会通过特定的输入诱导代理绕过原始的行为约束，然后触发某些有风险和敏感的操作，例如修改设置、执行未经授权的命令或冒充用户身份，这给系统安全带来了新的挑战。智能代理的现有安全措施在面临复杂的交互时仍然存在局限性，特别是在多轮对话或任务序列中检测潜在的危险行为方面。此外，目前还缺乏一种有效和一致的自动化方法来协助评估和确定这些风险的影响。本文探讨了移动多通道代理的安全问题，尝试通过引入行为序列信息构建风险鉴别机制，并设计了一种基于大型语言模型的自动辅助评估方案。通过在几个有代表性的高风险任务中的初步验证，结果表明该方法在一定程度上提高了对危险行为的识别，有助于降低智能体越狱的概率。希望本文的研究能为多通道智能代理系统的安全风险建模和防护提供一些有价值的参考。



## **15. CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs**

CAWLRY-V：一个用于视频MLLM对抗性攻击的大规模生成器框架 cs.CV

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00817v1) [paper-pdf](http://arxiv.org/pdf/2507.00817v1)

**Authors**: Jiaming Zhang, Rui Hu, Qing Guo, Wei Yang Bryan Lim

**Abstract**: Video Multimodal Large Language Models (V-MLLMs) have shown impressive capabilities in temporal reasoning and cross-modal understanding, yet their vulnerability to adversarial attacks remains underexplored due to unique challenges: complex cross-modal reasoning mechanisms, temporal dependencies, and computational constraints. We present CAVALRY-V (Cross-modal Language-Vision Adversarial Yielding for Videos), a novel framework that directly targets the critical interface between visual perception and language generation in V-MLLMs. Our approach introduces two key innovations: (1) a dual-objective semantic-visual loss function that simultaneously disrupts the model's text generation logits and visual representations to undermine cross-modal integration, and (2) a computationally efficient two-stage generator framework that combines large-scale pre-training for cross-model transferability with specialized fine-tuning for spatiotemporal coherence. Empirical evaluation on comprehensive video understanding benchmarks demonstrates that CAVALRY-V significantly outperforms existing attack methods, achieving 22.8% average improvement over the best baseline attacks on both commercial systems (GPT-4.1, Gemini 2.0) and open-source models (QwenVL-2.5, InternVL-2.5, Llava-Video, Aria, MiniCPM-o-2.6). Our framework achieves flexibility through implicit temporal coherence modeling rather than explicit regularization, enabling significant performance improvements even on image understanding (34.4% average gain). This capability demonstrates CAVALRY-V's potential as a foundational approach for adversarial research across multimodal systems.

摘要: 视频多模式大型语言模型（V-MLLM）在时态推理和跨模式理解方面表现出了令人印象深刻的能力，但由于独特的挑战，它们对对抗性攻击的脆弱性仍然没有得到充分的研究：复杂的跨模式推理机制、时态依赖性和计算限制。我们提出了CAWLRY-V（跨模式视觉对抗屈服视频），这是一个新颖的框架，直接针对V-MLLM中视觉感知和语言生成之间的关键界面。我们的方法引入了两个关键创新：（1）双目标语义视觉损失函数，它同时扰乱模型的文本生成日志和视觉表示以破坏跨模式集成，以及（2）计算高效的两阶段生成器框架，它将跨模型可移植性的大规模预训练与时空一致性的专门微调相结合。对全面视频理解基准的实证评估表明，CAWLRY-V的表现显着优于现有的攻击方法，比商业系统（GPT-4.1、Gemini 2.0）和开源模型（QwenVL-2.5、InternVL-2.5、Llava-Video、Aria、MiniCPM-o-2.6）的最佳基线攻击平均改进了22.8%。我们的框架通过隐式时间一致性建模而不是显式正规化来实现灵活性，即使在图像理解方面也能显着提高性能（平均提高34.4%）。这一能力展示了CAWLRY-V作为跨多模式系统对抗性研究的基础方法的潜力。



## **16. Impact of Fine-Tuning Methods on Memorization in Large Language Models**

微调方法对大型语言模型中精简化的影响 cs.CL

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2507.00258v1) [paper-pdf](http://arxiv.org/pdf/2507.00258v1)

**Authors**: Jie Hou, Chuxiong Wu, Lannan Luo, Qiang Zeng

**Abstract**: As the capabilities of pre-trained large language models (LLMs) continue to advance, the "pre-train and fine-tune" paradigm has become increasingly mainstream, leading to the development of various fine-tuning methods. However, the privacy risks arising from memorization during fine-tuning have received relatively little attention. To address this gap, we categorize popular fine-tuning approaches and assess their impact on memorization through the lens of membership inference attacks (MIAs). Our results show that, compared to parameter-based fine-tuning, prompt-based fine-tuning achieves competitive performance while exhibiting lower vulnerability to MIAs. Furthermore, prompt-based methods maintain low memorization regardless of model scale. These findings suggest that parameter-based fine-tuning is more prone to leaking private information, whereas prompt-based fine-tuning serves as a more privacy-preserving option.

摘要: 随着预训练大型语言模型（LLM）能力的不断进步，“预训练和微调”范式日益成为主流，导致各种微调方法的发展。然而，微调过程中因记忆而产生的隐私风险相对较少受到关注。为了解决这一差距，我们对流行的微调方法进行了分类，并通过成员资格推理攻击（MIA）的视角评估它们对记忆的影响。我们的结果表明，与基于参数的微调相比，基于预算的微调可以实现有竞争力的性能，同时对MIA的脆弱性更低。此外，无论模型规模如何，基于预算的方法都保持较低的记忆力。这些发现表明，基于参数的微调更容易泄露私人信息，而基于预算的微调则是一种更能保护隐私的选择。



## **17. Trust & Safety of LLMs and LLMs in Trust & Safety**

LLM的信任与安全以及LLM的信任与安全 cs.AI

11 pages

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2412.02113v2) [paper-pdf](http://arxiv.org/pdf/2412.02113v2)

**Authors**: Doohee You, Dan Chon

**Abstract**: In recent years, Large Language Models (LLMs) have garnered considerable attention for their remarkable abilities in natural language processing tasks. However, their widespread adoption has raised concerns pertaining to trust and safety. This systematic review investigates the current research landscape on trust and safety in LLMs, with a particular focus on the novel application of LLMs within the field of Trust and Safety itself. We delve into the complexities of utilizing LLMs in domains where maintaining trust and safety is paramount, offering a consolidated perspective on this emerging trend.\   By synthesizing findings from various studies, we identify key challenges and potential solutions, aiming to benefit researchers and practitioners seeking to understand the nuanced interplay between LLMs and Trust and Safety.   This review provides insights on best practices for using LLMs in Trust and Safety, and explores emerging risks such as prompt injection and jailbreak attacks. Ultimately, this study contributes to a deeper understanding of how LLMs can be effectively and responsibly utilized to enhance trust and safety in the digital realm.

摘要: 近年来，大型语言模型（LLM）因其在自然语言处理任务中的非凡能力而受到了广泛关注。然而，它们的广泛采用引发了人们对信任和安全的担忧。这篇系统性综述调查了当前关于LLM信任和安全的研究格局，特别关注LLM在信任和安全本身领域的新颖应用。我们深入研究了在维护信任和安全至关重要的领域中利用LLM的复杂性，为这一新兴趋势提供了统一的视角。\   通过综合各种研究的结果，我们确定了关键挑战和潜在的解决方案，旨在使寻求了解法学硕士与信任和安全之间微妙相互作用的研究人员和从业者受益。   本评论提供了有关在信任与安全中使用LLM的最佳实践的见解，并探讨了即时注射和越狱攻击等新出现的风险。最终，这项研究有助于更深入地了解如何有效、负责任地利用LLM来增强数字领域的信任和安全。



## **18. Logit-Gap Steering: Efficient Short-Suffix Jailbreaks for Aligned Large Language Models**

Logit-Gap Steering：Aligned Large Language Models的高效短后缀越狱 cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24056v1) [paper-pdf](http://arxiv.org/pdf/2506.24056v1)

**Authors**: Tung-Ling Li, Hongliang Liu

**Abstract**: We introduce logit-gap steering, a fast jailbreak framework that casts the refusal-affirmation gap of RLHF-aligned language models as a single pass over the vocabulary. A forward-computable score blends gap reduction with lightweight proxies for KL penalty and reward shift, allowing a "sort-sum-stop" sweep to complete in under a second and return a short suffix--two orders of magnitude fewer model calls than beam or gradient attacks. The same suffix generalises to unseen prompts and scales from 0.5 B to 70 B checkpoints, lifting one-shot attack success from baseline levels to 80-100% while preserving topical coherence. Beyond efficiency, these suffixes expose sentence-boundary reward cliffs and other alignment artefacts, offering a lightweight probe into how safety tuning reshapes internal representations.

摘要: 我们引入了logit-gap steering，这是一个快速越狱框架，它将RLHF对齐的语言模型的反思-肯定差距视为词汇的一次传递。可向前计算的分数将差距缩小与KL惩罚和奖励转移的轻量级代理结合起来，允许“排序和停止”扫描在一秒内完成并返回短后缀--模型调用比束或梯度攻击少两个数量级。相同的后缀推广到未见的提示，并将0.5 B到70 B检查点范围内，将一次性攻击成功率从基线水平提高到80-100%，同时保持话题连贯性。除了效率之外，这些后缀还暴露了行业边界奖励悬崖和其他对齐文物，为安全调整如何重塑内部表示提供了轻量级的探索。



## **19. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

谜语我！检索增强一代的隐形会员推断 cs.CR

This is the full version (27 pages) of the paper 'Riddle Me This!  Stealthy Membership Inference for Retrieval-Augmented Generation' published  at CCS 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2502.00306v2) [paper-pdf](http://arxiv.org/pdf/2502.00306v2)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.

摘要: 检索增强生成（RAG）使大型语言模型（LLM）能够通过利用外部知识数据库来生成接地响应，而无需更改模型参数。尽管缺乏权重调整可以防止模型参数泄露，但它引入了推理对手利用模型上下文中检索到的文档的风险。现有的隶属关系推断和数据提取方法通常依赖于越狱或精心制作的非自然查询，这些查询可以通过RAG系统中常见的查询重写技术轻松检测或阻止。在这项工作中，我们介绍了审讯攻击（IA），这是一种针对RAG收件箱中文档的成员资格推断技术。通过制作仅在目标文档存在的情况下才能回答的自然文本查询，我们的方法仅用30个查询就能证明成功推理，同时保持隐蔽性;简单的检测器识别来自现有方法的对抗性提示的频率高达约76倍，比我们的攻击产生的提示。我们观察到，在各种RAG配置中，TPR@1%FPR比之前的推理攻击提高了2倍，同时每个文档推理的成本不到0.02美元。



## **20. SoK: Semantic Privacy in Large Language Models**

SoK：大型语言模型中的语义隐私 cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23603v1) [paper-pdf](http://arxiv.org/pdf/2506.23603v1)

**Authors**: Baihe Ma, Yanna Jiang, Xu Wang, Guangshen Yu, Qin Wang, Caijun Sun, Chen Li, Xuelei Qi, Ying He, Wei Ni, Ren Ping Liu

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains, traditional data privacy measures prove inadequate for protecting information that is implicit, contextual, or inferable - what we define as semantic privacy. This Systematization of Knowledge (SoK) introduces a lifecycle-centric framework to analyze how semantic privacy risks emerge across input processing, pretraining, fine-tuning, and alignment stages of LLMs. We categorize key attack vectors and assess how current defenses, such as differential privacy, embedding encryption, edge computing, and unlearning, address these threats. Our analysis reveals critical gaps in semantic-level protection, especially against contextual inference and latent representation leakage. We conclude by outlining open challenges, including quantifying semantic leakage, protecting multimodal inputs, balancing de-identification with generation quality, and ensuring transparency in privacy enforcement. This work aims to inform future research on designing robust, semantically aware privacy-preserving techniques for LLMs.

摘要: 随着大型语言模型（LLM）越来越多地部署在敏感领域，传统的数据隐私措施被证明不足以保护隐性、上下文或可推理的信息--我们将其定义为语义隐私。该知识系统化（SoK）引入了一个以生命周期为中心的框架，以分析LLM的输入处理、预训练、微调和对齐阶段如何出现语义隐私风险。我们对关键攻击载体进行分类，并评估当前的防御措施（例如差异隐私、嵌入加密、边缘计算和取消学习）如何解决这些威胁。我们的分析揭示了语义级保护方面的关键差距，特别是针对上下文推断和潜在的表示泄露。最后，我们概述了开放的挑战，包括量化语义泄露、保护多模式输入、平衡去识别与生成质量以及确保隐私执行的透明度。这项工作旨在为未来关于为LLM设计稳健、语义感知的隐私保护技术的研究提供信息。



## **21. Evaluating Multi-Agent Defences Against Jailbreaking Attacks on Large Language Models**

评估针对大型语言模型越狱攻击的多智能体防御 cs.AI

26 pages, 1 figure

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23576v1) [paper-pdf](http://arxiv.org/pdf/2506.23576v1)

**Authors**: Maria Carolina Cornelia Wit, Jun Pang

**Abstract**: Recent advances in large language models (LLMs) have raised concerns about jailbreaking attacks, i.e., prompts that bypass safety mechanisms. This paper investigates the use of multi-agent LLM systems as a defence against such attacks. We evaluate three jailbreaking strategies, including the original AutoDefense attack and two from Deepleaps: BetterDan and JB. Reproducing the AutoDefense framework, we compare single-agent setups with two- and three-agent configurations. Our results show that multi-agent systems enhance resistance to jailbreaks, especially by reducing false negatives. However, its effectiveness varies by attack type, and it introduces trade-offs such as increased false positives and computational overhead. These findings point to the limitations of current automated defences and suggest directions for improving alignment robustness in future LLM systems.

摘要: 大型语言模型（LLM）的最新进展引发了对越狱攻击的担忧，即，绕过安全机制的提示。本文研究了使用多代理LLM系统作为防御这种攻击。我们评估了三种越狱策略，包括原始的AutoDefense攻击和Deepleaps的两种：BetterDan和JB。我们复制AutoDefense框架，比较了单代理设置与两个和三个代理配置。我们的结果表明，多智能体系统可以增强对越狱的抵抗力，特别是通过减少假阴性。然而，它的有效性因攻击类型而异，并且它引入了权衡，例如增加误报和计算负担。这些发现指出了当前自动化防御的局限性，并为提高未来LLM系统的对齐鲁棒性提出了方向。



## **22. TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs**

TuCo：衡量微调对LLM个人响应的贡献 cs.CL

ICML 2025

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23423v1) [paper-pdf](http://arxiv.org/pdf/2506.23423v1)

**Authors**: Felipe Nuti, Tim Franzmeyer, João Henriques

**Abstract**: Past work has studied the effects of fine-tuning on large language models' (LLMs) overall performance on certain tasks. However, a quantitative and systematic method for analyzing its effect on individual outputs is still lacking. Here, we propose a new method for measuring the contribution that fine-tuning makes to individual LLM responses, assuming access to the original pre-trained model. Our method tracks the model's intermediate hidden states, providing a more fine-grained insight into the effects of fine-tuning than a simple comparison of final outputs from pre-trained and fine-tuned models. We introduce and theoretically analyze an exact decomposition of any fine-tuned LLM into a pre-training component and a fine-tuning component. Empirically, we find that model behavior and performance can be steered by up- or down-scaling the fine-tuning component during the forward pass. Motivated by this finding and our theoretical analysis, we define the Tuning Contribution (TuCo) as the ratio of the magnitudes of the fine-tuning component to the pre-training component. We observe that three prominent adversarial attacks on LLMs circumvent safety measures in a way that reduces TuCo, and that TuCo is consistently lower on prompts where these attacks succeed compared to those where they do not. This suggests that attenuating the effect of fine-tuning on model outputs plays a role in the success of such attacks. In summary, TuCo enables the quantitative study of how fine-tuning influences model behavior and safety, and vice versa.

摘要: 过去的工作研究了微调对大型语言模型（LLM）在某些任务上整体性能的影响。然而，仍然缺乏一种定量、系统的方法来分析其对单个产出的影响。在这里，我们提出了一种新的方法来衡量微调对个体LLM响应的贡献，假设可以访问原始的预训练模型。我们的方法跟踪模型的中间隐藏状态，与预训练和微调模型的最终输出的简单比较相比，提供了对微调效果的更细粒度的见解。我们引入并从理论上分析将任何微调LLM精确分解为预训练组件和微调组件。从经验上看，我们发现模型行为和性能可以通过在前向传递期间放大或缩小微调组件来引导。受这一发现和理论分析的启发，我们将调整贡献（TuCo）定义为微调分量与预训练分量的幅度之比。我们观察到，针对LLM的三种突出的对抗性攻击以某种程度上减少了TuCo的方式规避了安全措施，并且与失败的情况相比，TuCo在这些攻击成功的提示上始终较低。这表明减弱微调对模型输出的影响在此类攻击的成功中发挥了作用。总之，TuCo能够定量研究微调如何影响模型行为和安全性，反之亦然。



## **23. Automating Adjudication of Cardiovascular Events Using Large Language Models**

使用大型语言模型自动判定心血管事件 cs.CL

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2503.17222v2) [paper-pdf](http://arxiv.org/pdf/2503.17222v2)

**Authors**: Sonish Sivarajkumar, Kimia Ameri, Chuqin Li, Yanshan Wang, Min Jiang

**Abstract**: Cardiovascular events, such as heart attacks and strokes, remain a leading cause of mortality globally, necessitating meticulous monitoring and adjudication in clinical trials. This process, traditionally performed manually by clinical experts, is time-consuming, resource-intensive, and prone to inter-reviewer variability, potentially introducing bias and hindering trial progress. This study addresses these critical limitations by presenting a novel framework for automating the adjudication of cardiovascular events in clinical trials using Large Language Models (LLMs). We developed a two-stage approach: first, employing an LLM-based pipeline for event information extraction from unstructured clinical data and second, using an LLM-based adjudication process guided by a Tree of Thoughts approach and clinical endpoint committee (CEC) guidelines. Using cardiovascular event-specific clinical trial data, the framework achieved an F1-score of 0.82 for event extraction and an accuracy of 0.68 for adjudication. Furthermore, we introduce the CLEART score, a novel, automated metric specifically designed for evaluating the quality of AI-generated clinical reasoning in adjudicating cardiovascular events. This approach demonstrates significant potential for substantially reducing adjudication time and costs while maintaining high-quality, consistent, and auditable outcomes in clinical trials. The reduced variability and enhanced standardization also allow for faster identification and mitigation of risks associated with cardiovascular therapies.

摘要: 心血管事件，如心脏病发作和中风，仍然是全球死亡的主要原因，需要在临床试验中进行细致的监测和裁定。这一过程传统上由临床专家手动执行，耗时，资源密集，并且容易出现审查员之间的差异，可能会引入偏倚并阻碍试验进展。本研究通过提出一种新的框架来解决这些关键的限制，该框架用于使用大型语言模型（LLM）自动裁定临床试验中的心血管事件。我们开发了一种两阶段方法：首先，采用基于LLM的管道从非结构化临床数据中提取事件信息，其次，使用基于LLM的裁定过程，由思想树方法和临床终点委员会（CEC）指南指导。使用心血管事件特定的临床试验数据，该框架的事件提取F1评分为0.82，裁定的准确性为0.68。此外，我们还引入了CREART评分，这是一种新型的自动化指标，专门用于评估裁定心血管事件时人工智能生成的临床推理的质量。这种方法在大幅减少裁定时间和成本的同时保持临床试验中的高质量、一致和可审计结果方面表现出巨大潜力。降低的变异性和增强的标准化还可以更快地识别和缓解与心血管治疗相关的风险。



## **24. Scaling Laws for Black box Adversarial Attacks**

黑匣子对抗攻击的缩放定律 cs.LG

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2411.16782v3) [paper-pdf](http://arxiv.org/pdf/2411.16782v3)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.

摘要: 对抗性示例通常表现出良好的跨模型可移植性，从而能够在有关其架构和参数的有限信息的情况下对黑匣子模型进行攻击，这在商业黑匣子场景中具有高度威胁性。模型集成是通过攻击多个代理模型来提高对抗性示例可移植性的有效策略。然而，由于之前的研究通常在整体中采用很少的模型，因此扩大模型数量是否可以进一步改善黑匣子攻击仍然是一个悬而未决的问题。受大型基金会模型缩放定律的启发，我们在这项工作中研究了黑匣子对抗攻击的缩放定律。通过理论分析和实证评估，我们得出了明确的缩放定律，即使用更多的代理模型增强了对抗性可转让性。全面的实验验证了标准图像分类器、多样化防御模型和使用各种对抗攻击方法的多模式大型语言模型的主张。具体来说，通过缩放定律，即使是GPT-4 o等专有模型，我们也能实现90%以上的传输攻击成功率。进一步的可视化表明，对抗性扰动的可解释性和语义也存在缩放定律。



## **25. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows**

从即时注入到协议漏洞：LLM支持的人工智能代理工作流程中的威胁 cs.CR

29 pages, 15 figures, 6 tables

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23260v1) [paper-pdf](http://arxiv.org/pdf/2506.23260v1)

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Djallel Hamouda, Leandros Maglaras, Merouane Debbah

**Abstract**: Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces have dramatically expanded capabilities for real-time data retrieval, complex computation, and multi-step orchestration. Yet, the explosive proliferation of plugins, connectors, and inter-agent protocols has outpaced discovery mechanisms and security practices, resulting in brittle integrations vulnerable to diverse threats. In this survey, we introduce the first unified, end-to-end threat model for LLM-agent ecosystems, spanning host-to-tool and agent-to-agent communications, formalize adversary capabilities and attacker objectives, and catalog over thirty attack techniques. Specifically, we organized the threat model into four domains: Input Manipulation (e.g., prompt injections, long-context hijacks, multimodal adversarial inputs), Model Compromise (e.g., prompt- and parameter-level backdoors, composite and encrypted multi-backdoors, poisoning strategies), System and Privacy Attacks (e.g., speculative side-channels, membership inference, retrieval poisoning, social-engineering simulations), and Protocol Vulnerabilities (e.g., exploits in Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent Network Protocol (ANP), and Agent-to-Agent (A2A) protocol). For each category, we review representative scenarios, assess real-world feasibility, and evaluate existing defenses. Building on our threat taxonomy, we identify key open challenges and future research directions, such as securing MCP deployments through dynamic trust management and cryptographic provenance tracking; designing and hardening Agentic Web Interfaces; and achieving resilience in multi-agent and federated environments. Our work provides a comprehensive reference to guide the design of robust defense mechanisms and establish best practices for resilient LLM-agent workflows.

摘要: 由具有结构化功能调用接口的大型语言模型（LLM）支持的自主人工智能代理极大地扩展了实时数据检索、复杂计算和多步骤编排的能力。然而，插件、连接器和代理间协议的爆炸性激增已经超过了发现机制和安全实践的速度，导致集成脆弱，容易受到各种威胁的影响。在本调查中，我们为LLM代理生态系统引入了第一个统一的端到端威胁模型，涵盖主机到工具和代理到代理的通信，正式化对手能力和攻击者目标，并对三十多种攻击技术进行了分类。具体来说，我们将威胁模型组织为四个领域：输入操纵（例如，提示注入、长上下文劫持、多模式对抗输入）、模型妥协（例如，提示和参数级后门、复合和加密的多后门、中毒策略）、系统和隐私攻击（例如，推测性侧通道、成员资格推断、检索中毒、社会工程模拟）和协议漏洞（例如，模型上下文协议（HCP）、代理通信协议（ACP）、代理网络协议（ANP）和代理对代理（A2 A）协议中的漏洞利用）。对于每个类别，我们都会审查代表性场景、评估现实世界的可行性并评估现有的防御措施。基于我们的威胁分类法，我们确定了关键的开放挑战和未来的研究方向，例如通过动态信任管理和加密来源跟踪来保护LCP部署;设计和强化统计Web界面;以及在多代理和联邦环境中实现弹性。我们的工作提供了全面的参考，以指导稳健的防御机制的设计并为弹性LLM代理工作流程建立最佳实践。



## **26. Guiding AI to Fix Its Own Flaws: An Empirical Study on LLM-Driven Secure Code Generation**

引导人工智能修复自身缺陷：LLM驱动的安全代码生成的实证研究 cs.SE

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.23034v1) [paper-pdf](http://arxiv.org/pdf/2506.23034v1)

**Authors**: Hao Yan, Swapneel Suhas Vaidya, Xiaokuan Zhang, Ziyu Yao

**Abstract**: Large Language Models (LLMs) have become powerful tools for automated code generation. However, these models often overlook critical security practices, which can result in the generation of insecure code that contains vulnerabilities-weaknesses or flaws in the code that attackers can exploit to compromise a system. However, there has been limited exploration of strategies to guide LLMs in generating secure code and a lack of in-depth analysis of the effectiveness of LLMs in repairing code containing vulnerabilities. In this paper, we present a comprehensive evaluation of state-of-the-art LLMs by examining their inherent tendencies to produce insecure code, their capability to generate secure code when guided by self-generated vulnerability hints, and their effectiveness in repairing vulnerabilities when provided with different levels of feedback. Our study covers both proprietary and open-weight models across various scales and leverages established benchmarks to assess a wide range of vulnerability types. Through quantitative and qualitative analyses, we reveal that although LLMs are prone to generating insecure code, advanced models can benefit from vulnerability hints and fine-grained feedback to avoid or fix vulnerabilities. We also provide actionable suggestions to developers to reduce vulnerabilities when using LLMs for code generation.

摘要: 大型语言模型（LLM）已成为自动代码生成的强大工具。然而，这些模型通常忽视了关键的安全实践，这可能会导致生成包含可操作性的不安全代码-攻击者可以利用代码中的弱点或缺陷来危害系统。然而，对指导LLM生成安全代码的策略的探索有限，并且缺乏对LLM修复包含漏洞的代码的有效性的深入分析。在本文中，我们通过检查它们产生不安全代码的固有倾向、它们在自我生成的漏洞提示的指导下生成安全代码的能力，以及它们在提供不同级别的反馈时修复漏洞的有效性，对最先进的LLM进行了全面评估。我们的研究涵盖了各种规模的专有模型和开放权重模型，并利用既定的基准来评估广泛的漏洞类型。通过定量和定性分析，我们发现，尽管LLM容易生成不安全的代码，但高级模型可以从漏洞提示和细粒度反馈中受益，以避免或修复漏洞。我们还向开发人员提供可操作的建议，以减少使用LLM生成代码时的漏洞。



## **27. Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models**

重温CroPA：视觉语言模型中交叉提示对抗可移植性的再现性研究和增强 cs.CV

Accepted to MLRC 2025

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22982v1) [paper-pdf](http://arxiv.org/pdf/2506.22982v1)

**Authors**: Atharv Mittal, Agam Pandey, Amritanshu Tiwari, Sukrit Jindal, Swadesh Swain

**Abstract**: Large Vision-Language Models (VLMs) have revolutionized computer vision, enabling tasks such as image classification, captioning, and visual question answering. However, they remain highly vulnerable to adversarial attacks, particularly in scenarios where both visual and textual modalities can be manipulated. In this study, we conduct a comprehensive reproducibility study of "An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models" validating the Cross-Prompt Attack (CroPA) and confirming its superior cross-prompt transferability compared to existing baselines. Beyond replication we propose several key improvements: (1) A novel initialization strategy that significantly improves Attack Success Rate (ASR). (2) Investigate cross-image transferability by learning universal perturbations. (3) A novel loss function targeting vision encoder attention mechanisms to improve generalization. Our evaluation across prominent VLMs -- including Flamingo, BLIP-2, and InstructBLIP as well as extended experiments on LLaVA validates the original results and demonstrates that our improvements consistently boost adversarial effectiveness. Our work reinforces the importance of studying adversarial vulnerabilities in VLMs and provides a more robust framework for generating transferable adversarial examples, with significant implications for understanding the security of VLMs in real-world applications.

摘要: 大型视觉语言模型（VLM）彻底改变了计算机视觉，实现了图像分类、字幕和视觉问答等任务。然而，它们仍然非常容易受到对抗攻击，特别是在视觉和文本模式都可以被操纵的场景中。在这项研究中，我们对“一个图像值得1000个谎言：视觉语言模型上的冲突可移植性”进行了全面的重复性研究，验证了交叉提示攻击（CroPA），并确认了与现有基线相比其优越的交叉提示可移植性。除了复制之外，我们还提出了几项关键改进：（1）一种新颖的初始化策略，可以显着提高攻击成功率（ASB）。(2)通过学习普适扰动来研究跨图像的可移植性。(3)一种针对视觉编码器注意力机制的新型损失函数，以提高概括性。我们对著名VLM（包括Flamingo、BLIP-2和INSTBLIP）的评估以及LLaVA的扩展实验验证了原始结果，并证明我们的改进持续提高了对抗有效性。我们的工作强调了研究VLM中对抗性漏洞的重要性，并为生成可转移的对抗性示例提供了一个更强大的框架，这对于理解现实世界应用程序中的VLM的安全性具有重要意义。



## **28. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

通过强化学习驱动的查询细化增强大型语言模型的能力和鲁棒性 cs.CL

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2407.01461v3) [paper-pdf](http://arxiv.org/pdf/2407.01461v3)

**Authors**: Xiaohua Wang, Zisu Huang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Qi Qian, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .

摘要: 大型语言模型（LLM）生成诚实、无害且有帮助的响应的能力严重依赖于用户提示的质量。然而，这些提示往往简短且模糊，从而严重限制了法学硕士的全部潜力。此外，对手可能会精心设计和操纵有害提示来越狱LLM，诱导它们产生潜在的有毒内容。为了增强LLM的能力，同时保持针对有害越狱输入的强大鲁棒性，本研究提出了一种可转移且可插入的框架，该框架在用户提示被输入LLM之前对其进行完善。该策略提高了查询的质量，使LLM能够生成更真实、良性和有用的响应。具体来说，使用专门设计的强化学习方法引入并训练轻量级查询细化模型，该方法结合了多个目标以增强LLM的特定能力。大量实验表明，细化模型不仅提高了响应的质量，而且增强了响应对越狱攻击的鲁棒性。代码可访问：https://github.com/Huangzisu/query-refinement。



## **29. Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation**

更小=更弱？量化LLM在代码生成中的基准测试鲁棒性 cs.SE

13 pages, 6 figures

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22776v1) [paper-pdf](http://arxiv.org/pdf/2506.22776v1)

**Authors**: Sen Fang, Weiyuan Ding, Antonio Mastropaolo, Bowen Xu

**Abstract**: Quantization has emerged as a mainstream method for compressing Large Language Models (LLMs), reducing memory requirements and accelerating inference without architectural modifications. While existing research primarily focuses on evaluating the effectiveness of quantized LLMs compared to their original counterparts, the impact on robustness remains largely unexplored.In this paper, we present the first systematic investigation of how quantization affects the robustness of LLMs in code generation tasks. Through extensive experiments across four prominent LLM families (LLaMA, DeepSeek, CodeGen, and StarCoder) with parameter scales ranging from 350M to 33B, we evaluate robustness from dual perspectives: adversarial attacks on input prompts and noise perturbations on model architecture. Our findings challenge conventional wisdom by demonstrating that quantized LLMs often exhibit superior robustness compared to their full-precision counterparts, with 51.59% versus 42.86% of our adversarial experiments showing better resilience in quantized LLMs. Similarly, our noise perturbation experiments also confirm that LLMs after quantitation generally withstand higher levels of weight disturbances. These results suggest that quantization not only reduces computational requirements but can actually enhance LLMs' reliability in code generation tasks, providing valuable insights for developing more robust and efficient LLM deployment strategies.

摘要: 量化已成为压缩大型语言模型（LLM）、减少内存需求并加速推理的主流方法，无需修改架构。虽然现有的研究主要集中在评估量化LLM与原始同类相比的有效性，但对稳健性的影响在很大程度上尚未探索。在本文中，我们首次系统地研究量化如何影响LLM在代码生成任务中的稳健性。通过对四个著名的LLM家族（LLaMA、DeepSeek、CodeGen和StarCoder）进行广泛实验，参数范围从350 M到33 B，我们从双重角度评估稳健性：对输入提示的对抗攻击和模型架构的噪音扰动。我们的研究结果挑战了传统智慧，证明量化LLM通常表现出比全精度同行更出色的鲁棒性，我们的对抗实验中分别有51.59%和42.86%表现出量化LLM更好的弹性。同样，我们的噪音扰动实验也证实，定量后的LLM通常可以承受更高水平的体重扰动。这些结果表明，量化不仅降低了计算要求，而且实际上可以增强LLM在代码生成任务中的可靠性，为开发更稳健、更高效的LLM部署策略提供有价值的见解。



## **30. MetaCipher: A General and Extensible Reinforcement Learning Framework for Obfuscation-Based Jailbreak Attacks on Black-Box LLMs**

MetaCipher：一个通用且可扩展的强化学习框架，用于对黑匣子LLM进行基于模糊的越狱攻击 cs.CR

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22557v1) [paper-pdf](http://arxiv.org/pdf/2506.22557v1)

**Authors**: Boyuan Chen, Minghao Shao, Abdul Basit, Siddharth Garg, Muhammad Shafique

**Abstract**: The growing capabilities of large language models (LLMs) have exposed them to increasingly sophisticated jailbreak attacks. Among these, obfuscation-based attacks -- which encrypt malicious content to evade detection -- remain highly effective. By leveraging the reasoning ability of advanced LLMs to interpret encrypted prompts, such attacks circumvent conventional defenses that rely on keyword detection or context filtering. These methods are very difficult to defend against, as existing safety mechanisms are not designed to interpret or decode ciphered content. In this work, we propose \textbf{MetaCipher}, a novel obfuscation-based jailbreak framework, along with a reinforcement learning-based dynamic cipher selection mechanism that adaptively chooses optimal encryption strategies from a cipher pool. This approach enhances jailbreak effectiveness and generalizability across diverse task types, victim LLMs, and safety guardrails. Our framework is modular and extensible by design, supporting arbitrary cipher families and accommodating evolving adversarial strategies. We complement our method with a large-scale empirical analysis of cipher performance across multiple victim LLMs. Within as few as 10 queries, MetaCipher achieves over 92\% attack success rate (ASR) on most recent standard malicious prompt benchmarks against state-of-the-art non-reasoning LLMs, and over 74\% ASR against reasoning-capable LLMs, outperforming all existing obfuscation-based jailbreak methods. These results highlight the long-term robustness and adaptability of our approach, making it more resilient than prior methods in the face of advancing safety measures.

摘要: 大型语言模型（LLM）不断增长的能力使它们面临越来越复杂的越狱攻击。其中，基于模糊的攻击（对恶意内容进行加密以逃避检测）仍然非常有效。通过利用高级LLM的推理能力来解释加密提示，此类攻击绕过了依赖关键字检测或上下文过滤的传统防御措施。这些方法非常难以防御，因为现有的安全机制不是为了解释或解码加密内容而设计的。在这项工作中，我们提出了\textBF{MetaCipher}，这是一种新型的基于模糊的越狱框架，以及一种基于强化学习的动态密码选择机制，该机制从密码池中自适应地选择最佳加密策略。这种方法增强了不同任务类型、受害者LLM和安全护栏的越狱有效性和普遍性。我们的框架是模块化的，可通过设计扩展，支持任意密码族并适应不断发展的对抗策略。我们通过对多个受害LLM的密码性能进行大规模实证分析来补充我们的方法。在短短10个查询内，MetaCipher针对最新的非推理LLM在最新标准恶意提示基准上就达到了超过92%的攻击成功率（ASB），针对具有推理能力的LLM达到了超过74%的攻击成功率，优于所有现有的基于模糊的越狱方法。这些结果凸显了我们方法的长期稳健性和适应性，使其在面对先进的安全措施时比以前的方法更具弹性。



## **31. Design Patterns for Securing LLM Agents against Prompt Injections**

保护LLM代理免受即时注射的设计模式 cs.LG

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.08837v3) [paper-pdf](http://arxiv.org/pdf/2506.08837v3)

**Authors**: Luca Beurer-Kellner, Beat Buesser, Ana-Maria Creţu, Edoardo Debenedetti, Daniel Dobos, Daniel Fabian, Marc Fischer, David Froelicher, Kathrin Grosse, Daniel Naeff, Ezinwanne Ozoani, Andrew Paverd, Florian Tramèr, Václav Volhejn

**Abstract**: As AI agents powered by Large Language Models (LLMs) become increasingly versatile and capable of addressing a broad spectrum of tasks, ensuring their security has become a critical challenge. Among the most pressing threats are prompt injection attacks, which exploit the agent's resilience on natural language inputs -- an especially dangerous threat when agents are granted tool access or handle sensitive information. In this work, we propose a set of principled design patterns for building AI agents with provable resistance to prompt injection. We systematically analyze these patterns, discuss their trade-offs in terms of utility and security, and illustrate their real-world applicability through a series of case studies.

摘要: 随着由大型语言模型（LLM）支持的AI代理变得越来越多才多艺，能够解决广泛的任务，确保其安全性已成为一项关键挑战。最紧迫的威胁之一是即时注入攻击，它利用代理对自然语言输入的弹性-当代理被授予工具访问或处理敏感信息时，这是一个特别危险的威胁。在这项工作中，我们提出了一套原则性的设计模式，用于构建具有可证明的即时注入阻力的AI代理。我们系统地分析了这些模式，讨论了它们在实用性和安全性方面的权衡，并通过一系列案例研究说明了它们在现实世界中的适用性。



## **32. Jailbreaking Multimodal Large Language Models via Shuffle Inconsistency**

通过洗牌不一致性破解多模式大型语言模型 cs.CR

ICCV2025

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2501.04931v2) [paper-pdf](http://arxiv.org/pdf/2501.04931v2)

**Authors**: Shiji Zhao, Ranjie Duan, Fengxiang Wang, Chi Chen, Caixin Kang, Shouwei Ruan, Jialing Tao, YueFeng Chen, Hui Xue, Xingxing Wei

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive performance and have been put into practical use in commercial applications, but they still have potential safety mechanism vulnerabilities. Jailbreak attacks are red teaming methods that aim to bypass safety mechanisms and discover MLLMs' potential risks. Existing MLLMs' jailbreak methods often bypass the model's safety mechanism through complex optimization methods or carefully designed image and text prompts. Despite achieving some progress, they have a low attack success rate on commercial closed-source MLLMs. Unlike previous research, we empirically find that there exists a Shuffle Inconsistency between MLLMs' comprehension ability and safety ability for the shuffled harmful instruction. That is, from the perspective of comprehension ability, MLLMs can understand the shuffled harmful text-image instructions well. However, they can be easily bypassed by the shuffled harmful instructions from the perspective of safety ability, leading to harmful responses. Then we innovatively propose a text-image jailbreak attack named SI-Attack. Specifically, to fully utilize the Shuffle Inconsistency and overcome the shuffle randomness, we apply a query-based black-box optimization method to select the most harmful shuffled inputs based on the feedback of the toxic judge model. A series of experiments show that SI-Attack can improve the attack's performance on three benchmarks. In particular, SI-Attack can obviously improve the attack success rate for commercial MLLMs such as GPT-4o or Claude-3.5-Sonnet.

摘要: 多模式大型语言模型（MLLM）取得了令人印象深刻的性能，并已在商业应用中投入实际使用，但它们仍然存在潜在的安全机制漏洞。越狱攻击是一种红色团队方法，旨在绕过安全机制并发现MLLM的潜在风险。现有的MLLM越狱方法通常通过复杂的优化方法或精心设计的图像和文本提示绕过模型的安全机制。尽管取得了一些进展，但他们对商业闭源MLLM的攻击成功率很低。与之前的研究不同，我们通过经验发现，MLLM对洗牌后的有害指令的理解能力和安全能力之间存在洗牌不一致性。也就是说，从理解能力的角度来看，MLLM能够很好地理解洗牌后的有害文本图像指令。然而，从安全能力的角度来看，它们很容易被洗牌的有害指令绕过，导致有害反应。然后我们创新性地提出了一种名为SI-Attack的文本图像越狱攻击。具体来说，为了充分利用洗牌不一致性并克服洗牌随机性，我们应用基于查询的黑匣子优化方法根据有毒判断模型的反馈选择最有害的洗牌输入。一系列实验表明，SI-Attack可以在三个基准测试上提高攻击的性能。特别是，SI-Attack可以明显提高GPT-4 o或Claude-3.5-Sonnet等商业MLLM的攻击成功率。



## **33. Cannot See the Forest for the Trees: Invoking Heuristics and Biases to Elicit Irrational Choices of LLMs**

只见树木不见森林：利用启发式和偏见来激发对LLM的非理性选择 cs.CL

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2505.02862v3) [paper-pdf](http://arxiv.org/pdf/2505.02862v3)

**Authors**: Haoming Yang, Ke Ma, Xiaojun Jia, Yingfei Sun, Qianqian Xu, Qingming Huang

**Abstract**: Despite the remarkable performance of Large Language Models (LLMs), they remain vulnerable to jailbreak attacks, which can compromise their safety mechanisms. Existing studies often rely on brute-force optimization or manual design, failing to uncover potential risks in real-world scenarios. To address this, we propose a novel jailbreak attack framework, ICRT, inspired by heuristics and biases in human cognition. Leveraging the simplicity effect, we employ cognitive decomposition to reduce the complexity of malicious prompts. Simultaneously, relevance bias is utilized to reorganize prompts, enhancing semantic alignment and inducing harmful outputs effectively. Furthermore, we introduce a ranking-based harmfulness evaluation metric that surpasses the traditional binary success-or-failure paradigm by employing ranking aggregation methods such as Elo, HodgeRank, and Rank Centrality to comprehensively quantify the harmfulness of generated content. Experimental results show that our approach consistently bypasses mainstream LLMs' safety mechanisms and generates high-risk content, providing insights into jailbreak attack risks and contributing to stronger defense strategies.

摘要: 尽管大型语言模型（LLM）性能出色，但它们仍然容易受到越狱攻击，这可能会损害其安全机制。现有的研究通常依赖于暴力优化或手动设计，未能发现现实世界场景中的潜在风险。为了解决这个问题，我们提出了一种新颖的越狱攻击框架ICRT，其灵感来自人类认知中的启发和偏见。利用简单性效应，我们采用认知分解来降低恶意提示的复杂性。同时，利用相关性偏差来重组提示，增强语义对齐并有效地诱导有害输出。此外，我们引入了一种基于排名的危害性评估指标，通过采用Elo、HodgeRank和Rank Centrality等排名聚合方法来全面量化生成内容的危害性，超越了传统的二元成败范式。实验结果表明，我们的方法始终绕过主流LLM的安全机制并生成高风险内容，提供了对越狱攻击风险的见解，并有助于制定更强有力的防御策略。



## **34. STAIR: Improving Safety Alignment with Introspective Reasoning**

楼梯：通过内省推理改善安全性 cs.CL

22 pages, 8 figures, ICML2025 Oral

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2502.02384v2) [paper-pdf](http://arxiv.org/pdf/2502.02384v2)

**Authors**: Yichi Zhang, Siyuan Zhang, Yao Huang, Zeyu Xia, Zhengwei Fang, Xiao Yang, Ranjie Duan, Dong Yan, Yinpeng Dong, Jun Zhu

**Abstract**: Ensuring the safety and harmlessness of Large Language Models (LLMs) has become equally critical as their performance in applications. However, existing safety alignment methods typically suffer from safety-performance trade-offs and the susceptibility to jailbreak attacks, primarily due to their reliance on direct refusals for malicious queries. In this paper, we propose STAIR, a novel framework that integrates SafeTy Alignment with Itrospective Reasoning. We enable LLMs to identify safety risks through step-by-step analysis by self-improving chain-of-thought (CoT) reasoning with safety awareness. STAIR first equips the model with a structured reasoning capability and then advances safety alignment via iterative preference optimization on step-level reasoning data generated using our newly proposed Safety-Informed Monte Carlo Tree Search (SI-MCTS). We further train a process reward model on this data to guide test-time searches for improved responses. Extensive experiments show that STAIR effectively mitigates harmful outputs while better preserving helpfulness, compared to instinctive alignment strategies. With test-time scaling, STAIR achieves a safety performance comparable to Claude-3.5 against popular jailbreak attacks. Relevant resources in this work are available at https://github.com/thu-ml/STAIR.

摘要: 确保大型语言模型（LLM）的安全性和无害性与其在应用程序中的性能一样重要。然而，现有的安全对齐方法通常会面临安全性能权衡和越狱攻击的易感性，这主要是由于它们依赖于直接拒绝恶意查询。在本文中，我们提出了STAIR，这是一个将SafeTy对齐与前瞻性推理集成的新型框架。我们使LLM能够通过具有安全意识的自我改进思维链（CoT）推理，通过逐步分析来识别安全风险。STAIR首先为模型配备结构化推理能力，然后通过对使用我们新提出的安全知情蒙特卡洛树搜索（SI-MCTS）生成的分步推理数据进行迭代偏好优化来推进安全对齐。我们根据这些数据进一步训练过程奖励模型，以指导测试时搜索以获得更好的响应。大量实验表明，与本能的对齐策略相比，STair可以有效地减轻有害输出，同时更好地保留帮助性。通过测试时间扩展，STAIR在对抗流行越狱攻击时实现了与Claude-3.5相当的安全性能。本作品的相关资源可访问https://github.com/thu-ml/STAIR。



## **35. Advancing Jailbreak Strategies: A Hybrid Approach to Exploiting LLM Vulnerabilities and Bypassing Modern Defenses**

推进越狱策略：利用LLM漏洞和扩展现代防御的混合方法 cs.CL

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21972v1) [paper-pdf](http://arxiv.org/pdf/2506.21972v1)

**Authors**: Mohamed Ahmed, Mohamed Abdelmouty, Mingyu Kim, Gunvanth Kandula, Alex Park, James C. Davis

**Abstract**: The advancement of Pre-Trained Language Models (PTLMs) and Large Language Models (LLMs) has led to their widespread adoption across diverse applications. Despite their success, these models remain vulnerable to attacks that exploit their inherent weaknesses to bypass safety measures. Two primary inference-phase threats are token-level and prompt-level jailbreaks. Token-level attacks embed adversarial sequences that transfer well to black-box models like GPT but leave detectable patterns and rely on gradient-based token optimization, whereas prompt-level attacks use semantically structured inputs to elicit harmful responses yet depend on iterative feedback that can be unreliable. To address the complementary limitations of these methods, we propose two hybrid approaches that integrate token- and prompt-level techniques to enhance jailbreak effectiveness across diverse PTLMs. GCG + PAIR and the newly explored GCG + WordGame hybrids were evaluated across multiple Vicuna and Llama models. GCG + PAIR consistently raised attack-success rates over its constituent techniques on undefended models; for instance, on Llama-3, its Attack Success Rate (ASR) reached 91.6%, a substantial increase from PAIR's 58.4% baseline. Meanwhile, GCG + WordGame matched the raw performance of WordGame maintaining a high ASR of over 80% even under stricter evaluators like Mistral-Sorry-Bench. Crucially, both hybrids retained transferability and reliably pierced advanced defenses such as Gradient Cuff and JBShield, which fully blocked single-mode attacks. These findings expose previously unreported vulnerabilities in current safety stacks, highlight trade-offs between raw success and defensive robustness, and underscore the need for holistic safeguards against adaptive adversaries.

摘要: 预训练语言模型（PTLM）和大型语言模型（LLM）的进步导致它们在不同的应用程序中广泛采用。尽管取得了成功，但这些模型仍然容易受到利用其固有弱点绕过安全措施的攻击。两种主要的推理阶段威胁是代币级和预算级越狱。令牌级攻击嵌入对抗序列，这些序列可以很好地传输到GPT等黑匣子模型，但留下可检测的模式并依赖于基于梯度的令牌优化，而预算级攻击使用语义结构化的输入来引发有害响应，但依赖于可能不可靠的迭代反馈。为了解决这些方法的互补局限性，我们提出了两种混合方法，集成代币和预算级技术，以增强不同PTLM之间的越狱有效性。GCG + PAIR和新探索的GCG + WordGame混合体在多个Vicuna和Lama模型中进行了评估。GCG + PAIR在无防御模型上始终提高了其组成技术的攻击成功率;例如，在Lama-3上，其攻击成功率（ASB）达到91.6%，比PAIR的58.4%基线大幅提高。与此同时，GCG + WordGame与WordGame的原始表现相媲美，即使在Mistral-Sorry-Bench等更严格的评估者下，也保持了超过80%的高ASB。至关重要的是，这两种混合体都保留了可转移性，并可靠地突破了Gradient Cuff和JB Shield等先进防御，从而完全阻止了单一模式攻击。这些发现暴露了当前安全堆栈中以前未报告的漏洞，强调了原始成功和防御稳健性之间的权衡，并强调了针对适应性对手的全面保障措施的必要性。



## **36. Exploring Task-Solving Paradigm for Generalized Cross-Domain Face Anti-Spoofing via Reinforcement Fine-Tuning**

通过强化微调探索广义跨域人脸反欺骗的任务求解范式 cs.CV

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21895v1) [paper-pdf](http://arxiv.org/pdf/2506.21895v1)

**Authors**: Fangling Jiang, Qi Li, Weining Wang, Gang Wang, Bing Liu, Zhenan Sun

**Abstract**: Recently the emergence of novel presentation attacks has drawn increasing attention to face anti-spoofing. However, existing methods tend to memorize data patterns from the training set, resulting in poor generalization to unknown attack types across different scenarios and limited interpretability. To address these challenges, this paper presents a reinforcement fine-tuning-based face anti-spoofing method that stimulates the capabilities of multimodal large language models to think and learn how to solve the anti-spoofing task itself, rather than relying on the memorization of authenticity patterns. We design verifiable class consistent reward and reasoning consistent reward, and employ a GRPO-based optimization strategy to guide the model in exploring reasoning policies from multiple perspectives to maximize expected rewards. As a result, through iterative trial-and-error learning while retaining only high-reward trajectories, the model distills highly generalizable decision-making rules from the extensive solution space to effectively address cross-domain face anti-spoofing tasks. Extensive experimental results demonstrate that our method achieves state-of-the-art cross-domain generalization performance. It generalizes well to diverse unknown attack types in unseen target domains while providing interpretable reasoning for its authenticity decisions without requiring labor-intensive textual annotations for training.

摘要: 最近，新颖的演示攻击的出现引起了人们对面部反欺骗的越来越多的关注。然而，现有的方法往往会从训练集中记住数据模式，导致对不同场景中未知攻击类型的概括性较差，并且解释性有限。为了应对这些挑战，本文提出了一种基于强化微调的面部反欺骗方法，该方法激发多模式大型语言模型思考和学习如何解决反欺骗任务本身的能力，而不是依赖于真实性模式的记忆。我们设计了可验证的类一致性奖励和推理一致性奖励，并采用基于GRPO的优化策略来指导模型从多个角度探索推理策略，以最大化预期奖励。因此，通过迭代试错学习，同时仅保留高回报轨迹，该模型从广泛的解决方案空间中提炼出高度可概括的决策规则，以有效地解决跨域面部反欺骗任务。大量的实验结果表明，我们的方法实现了最先进的跨域概括性能。它很好地推广到不可见目标领域中的各种未知攻击类型，同时为其真实性决策提供可解释的推理，而无需劳动密集型文本注释进行训练。



## **37. $C^3$-Bench: The Things Real Disturbing LLM based Agent in Multi-Tasking**

$C ' 3 $-Bench：多任务中令人不安的LLM代理人真正不安的事情 cs.AI

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2505.18746v4) [paper-pdf](http://arxiv.org/pdf/2505.18746v4)

**Authors**: Peijie Yu, Yifan Yang, Jinjian Li, Zelong Zhang, Haorui Wang, Xiao Feng, Feng Zhang

**Abstract**: Agents based on large language models leverage tools to modify environments, revolutionizing how AI interacts with the physical world. Unlike traditional NLP tasks that rely solely on historical dialogue for responses, these agents must consider more complex factors, such as inter-tool relationships, environmental feedback and previous decisions, when making choices. Current research typically evaluates agents via multi-turn dialogues. However, it overlooks the influence of these critical factors on agent behavior. To bridge this gap, we present an open-source and high-quality benchmark $C^3$-Bench. This benchmark integrates attack concepts and applies univariate analysis to pinpoint key elements affecting agent robustness. In concrete, we design three challenges: navigate complex tool relationships, handle critical hidden information and manage dynamic decision paths. Complementing these challenges, we introduce fine-grained metrics, innovative data collection algorithms and reproducible evaluation methods. Extensive experiments are conducted on 49 mainstream agents, encompassing general fast-thinking, slow-thinking and domain-specific models. We observe that agents have significant shortcomings in handling tool dependencies, long context information dependencies and frequent policy-type switching. In essence, $C^3$-Bench aims to expose model vulnerabilities through these challenges and drive research into the interpretability of agent performance. The benchmark is publicly available at https://github.com/TencentHunyuan/C3-Benchmark.

摘要: 基于大型语言模型的代理利用工具来修改环境，彻底改变了人工智能与物理世界交互的方式。与仅依赖历史对话来做出反应的传统NLP任务不同，这些代理人在做出选择时必须考虑更复杂的因素，例如工具间关系、环境反馈和之前的决策。当前的研究通常通过多轮对话来评估代理人。然而，它忽略了这些关键因素对代理行为的影响。为了弥合这一差距，我们提出了一个开源且高质量的基准$C#3 $-Bench。该基准测试集成了攻击概念并应用单变量分析来确定影响代理稳健性的关键元素。具体而言，我们设计了三个挑战：导航复杂的工具关系、处理关键的隐藏信息以及管理动态决策路径。为了补充这些挑战，我们引入了细粒度指标、创新的数据收集算法和可重复的评估方法。对49种主流代理进行了广泛的实验，涵盖了一般快速思维、缓慢思维和特定领域的模型。我们观察到，代理在处理工具依赖性、长上下文信息依赖性和频繁的策略类型切换方面存在显着缺陷。本质上，$C^3$-Bench旨在通过这些挑战暴露模型漏洞，并推动对代理性能可解释性的研究。该基准可在https://github.com/TencentHunyuan/C3-Benchmark上公开获得。



## **38. On the Feasibility of Poisoning Text-to-Image AI Models via Adversarial Mislabeling**

论通过对抗性错误标签毒害文本到图像人工智能模型的可行性 cs.CR

ACM Conference on Computer and Communications Security 2025

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21874v1) [paper-pdf](http://arxiv.org/pdf/2506.21874v1)

**Authors**: Stanley Wu, Ronik Bhaskar, Anna Yoo Jeong Ha, Shawn Shan, Haitao Zheng, Ben Y. Zhao

**Abstract**: Today's text-to-image generative models are trained on millions of images sourced from the Internet, each paired with a detailed caption produced by Vision-Language Models (VLMs). This part of the training pipeline is critical for supplying the models with large volumes of high-quality image-caption pairs during training. However, recent work suggests that VLMs are vulnerable to stealthy adversarial attacks, where adversarial perturbations are added to images to mislead the VLMs into producing incorrect captions.   In this paper, we explore the feasibility of adversarial mislabeling attacks on VLMs as a mechanism to poisoning training pipelines for text-to-image models. Our experiments demonstrate that VLMs are highly vulnerable to adversarial perturbations, allowing attackers to produce benign-looking images that are consistently miscaptioned by the VLM models. This has the effect of injecting strong "dirty-label" poison samples into the training pipeline for text-to-image models, successfully altering their behavior with a small number of poisoned samples. We find that while potential defenses can be effective, they can be targeted and circumvented by adaptive attackers. This suggests a cat-and-mouse game that is likely to reduce the quality of training data and increase the cost of text-to-image model development. Finally, we demonstrate the real-world effectiveness of these attacks, achieving high attack success (over 73%) even in black-box scenarios against commercial VLMs (Google Vertex AI and Microsoft Azure).

摘要: 当今的文本到图像生成模型是在来自互联网的数百万张图像上训练的，每个图像都与视觉语言模型（VLM）生成的详细标题配对。训练管道的这一部分对于在训练期间为模型提供大量高质量图像字幕对至关重要。然而，最近的研究表明，VLM很容易受到隐蔽的对抗攻击，对抗性扰动被添加到图像中以误导VLM产生错误的字幕。   本文中，我们探讨了对VLM的对抗性错误标签攻击作为毒害文本到图像模型训练管道的机制的可行性。我们的实验表明，VLM非常容易受到对抗性扰动的影响，这使得攻击者能够生成看似友善的图像，而这些图像始终被VLM模型字幕错误。这的效果是将强“肮脏标签”毒物样本注入文本到图像模型的训练管道中，用少量毒物样本成功改变它们的行为。我们发现，虽然潜在的防御措施可能有效，但它们可能会被适应性攻击者瞄准和规避。这表明猫鼠游戏可能会降低训练数据的质量并增加文本到图像模型开发的成本。最后，我们展示了这些攻击在现实世界中的有效性，即使在针对商业VLM（Google Vertex AI和Microsoft Azure）的黑匣子场景中，也实现了很高的攻击成功率（超过73%）。



## **39. A Survey on Model Extraction Attacks and Defenses for Large Language Models**

大型语言模型的模型提取攻击和防御综述 cs.CR

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.22521v1) [paper-pdf](http://arxiv.org/pdf/2506.22521v1)

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong

**Abstract**: Model extraction attacks pose significant security threats to deployed language models, potentially compromising intellectual property and user privacy. This survey provides a comprehensive taxonomy of LLM-specific extraction attacks and defenses, categorizing attacks into functionality extraction, training data extraction, and prompt-targeted attacks. We analyze various attack methodologies including API-based knowledge distillation, direct querying, parameter recovery, and prompt stealing techniques that exploit transformer architectures. We then examine defense mechanisms organized into model protection, data privacy protection, and prompt-targeted strategies, evaluating their effectiveness across different deployment scenarios. We propose specialized metrics for evaluating both attack effectiveness and defense performance, addressing the specific challenges of generative language models. Through our analysis, we identify critical limitations in current approaches and propose promising research directions, including integrated attack methodologies and adaptive defense mechanisms that balance security with model utility. This work serves NLP researchers, ML engineers, and security professionals seeking to protect language models in production environments.

摘要: 模型提取攻击对已部署的语言模型构成重大安全威胁，可能会损害知识产权和用户隐私。这项调查提供了LLM特定的提取攻击和防御的全面分类，将攻击分为功能提取、训练数据提取和预算目标攻击。我们分析了各种攻击方法，包括基于API的知识蒸馏，直接查询，参数恢复，并迅速窃取技术，利用Transformer架构。然后，我们检查了分为模型保护、数据隐私保护和预算目标策略在内的防御机制，评估它们在不同部署场景中的有效性。我们提出了评估攻击有效性和防御性能的专门指标，以解决生成式语言模型的具体挑战。通过我们的分析，我们发现了当前方法的关键局限性，并提出了有前途的研究方向，包括平衡安全性与模型效用的集成攻击方法和自适应防御机制。这项工作为NLP研究人员、ML工程师和寻求保护生产环境中的语言模型的安全专业人员提供服务。



## **40. Domain Knowledge-Enhanced LLMs for Fraud and Concept Drift Detection**

用于欺诈和概念漂移检测的领域知识增强型LLM cs.CL

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.21443v1) [paper-pdf](http://arxiv.org/pdf/2506.21443v1)

**Authors**: Ali Şenol, Garima Agrawal, Huan Liu

**Abstract**: Detecting deceptive conversations on dynamic platforms is increasingly difficult due to evolving language patterns and Concept Drift (CD)-i.e., semantic or topical shifts that alter the context or intent of interactions over time. These shifts can obscure malicious intent or mimic normal dialogue, making accurate classification challenging. While Large Language Models (LLMs) show strong performance in natural language tasks, they often struggle with contextual ambiguity and hallucinations in risk-sensitive scenarios. To address these challenges, we present a Domain Knowledge (DK)-Enhanced LLM framework that integrates pretrained LLMs with structured, task-specific insights to perform fraud and concept drift detection. The proposed architecture consists of three main components: (1) a DK-LLM module to detect fake or deceptive conversations; (2) a drift detection unit (OCDD) to determine whether a semantic shift has occurred; and (3) a second DK-LLM module to classify the drift as either benign or fraudulent. We first validate the value of domain knowledge using a fake review dataset and then apply our full framework to SEConvo, a multiturn dialogue dataset that includes various types of fraud and spam attacks. Results show that our system detects fake conversations with high accuracy and effectively classifies the nature of drift. Guided by structured prompts, the LLaMA-based implementation achieves 98% classification accuracy. Comparative studies against zero-shot baselines demonstrate that incorporating domain knowledge and drift awareness significantly improves performance, interpretability, and robustness in high-stakes NLP applications.

摘要: 由于不断演变的语言模式和概念漂移（CD），在动态平台上检测欺骗性对话越来越困难，即，随着时间的推移改变交互的上下文或意图的语义或话题转变。这些转变可能会掩盖恶意意图或模仿正常对话，从而使准确的分类具有挑战性。虽然大型语言模型（LLM）在自然语言任务中表现出很强的性能，但在风险敏感的场景中，它们往往会遇到上下文模糊和幻觉。为了应对这些挑战，我们提出了一个领域知识（DK）-增强型LLM框架，该框架将预训练的LLM与结构化的、特定于任务的洞察集成起来，以执行欺诈和概念漂移检测。所提出的架构由三个主要组件组成：（1）DK-LLM模块，用于检测虚假或欺骗性对话;（2）漂移检测单元（ODDD），用于确定是否发生了语义转变;（3）第二DK-LLM模块，用于将漂移分类为良性或欺诈性。我们首先使用虚假审查数据集验证领域知识的价值，然后将我们的完整框架应用于SEConvo，这是一个多回合对话数据集，包括各种类型的欺诈和垃圾邮件攻击。结果表明，我们的系统能够高准确性地检测虚假对话，并有效地对漂移的性质进行分类。在结构化提示的指导下，基于LLaMA的实现实现了98%的分类准确率。与零触发基线的比较研究表明，结合领域知识和漂移意识可以显着提高高风险NLP应用程序的性能、可解释性和鲁棒性。



## **41. TracLLM: A Generic Framework for Attributing Long Context LLMs**

TracLLM：用于赋予长上下文LLM属性的通用框架 cs.CR

To appear in USENIX Security Symposium 2025. The code and data are  at: https://github.com/Wang-Yanting/TracLLM

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.04202v3) [paper-pdf](http://arxiv.org/pdf/2506.04202v3)

**Authors**: Yanting Wang, Wei Zou, Runpeng Geng, Jinyuan Jia

**Abstract**: Long context large language models (LLMs) are deployed in many real-world applications such as RAG, agent, and broad LLM-integrated applications. Given an instruction and a long context (e.g., documents, PDF files, webpages), a long context LLM can generate an output grounded in the provided context, aiming to provide more accurate, up-to-date, and verifiable outputs while reducing hallucinations and unsupported claims. This raises a research question: how to pinpoint the texts (e.g., sentences, passages, or paragraphs) in the context that contribute most to or are responsible for the generated output by an LLM? This process, which we call context traceback, has various real-world applications, such as 1) debugging LLM-based systems, 2) conducting post-attack forensic analysis for attacks (e.g., prompt injection attack, knowledge corruption attacks) to an LLM, and 3) highlighting knowledge sources to enhance the trust of users towards outputs generated by LLMs. When applied to context traceback for long context LLMs, existing feature attribution methods such as Shapley have sub-optimal performance and/or incur a large computational cost. In this work, we develop TracLLM, the first generic context traceback framework tailored to long context LLMs. Our framework can improve the effectiveness and efficiency of existing feature attribution methods. To improve the efficiency, we develop an informed search based algorithm in TracLLM. We also develop contribution score ensemble/denoising techniques to improve the accuracy of TracLLM. Our evaluation results show TracLLM can effectively identify texts in a long context that lead to the output of an LLM. Our code and data are at: https://github.com/Wang-Yanting/TracLLM.

摘要: 长上下文大型语言模型（LLM）部署在许多现实世界的应用程序中，例如RAG、代理和广泛的LLM集成应用程序。给定指令和长上下文（例如，文档、PDF文件、网页）、长上下文LLM可以基于所提供的上下文生成输出，旨在提供更准确、最新和可验证的输出，同时减少幻觉和不支持的主张。这提出了一个研究问题：如何确定文本（例如，句子、段落或段落）在对LLM生成的输出做出最大贡献或负责的上下文中？这个过程（我们称之为上下文追溯）具有各种现实世界的应用程序，例如1）调试基于LLM的系统，2）对攻击进行攻击后取证分析（例如，即时注入攻击、知识腐败攻击）对LLM，以及3）强调知识源以增强用户对LLM生成的输出的信任。当应用于长上下文LLM的上下文追溯时，现有的特征属性方法（例如Shapley）的性能不佳和/或会产生很大的计算成本。在这项工作中，我们开发了TracLLM，这是第一个针对长上下文LLM量身定制的通用上下文追溯框架。我们的框架可以提高现有特征归因方法的有效性和效率。为了提高效率，我们在TracLLM中开发了一种基于明智搜索的算法。我们还开发贡献分数集成/去噪技术来提高TracLLM的准确性。我们的评估结果表明TracLLM可以有效地识别导致LLM输出的长期上下文中的文本。我们的代码和数据位于：https://github.com/Wang-Yanting/TracLLM。



## **42. A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns**

具有传染性的越狱麻烦制造者扰乱诚实城镇 cs.CL

ACL 2025 Main

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2410.16155v2) [paper-pdf](http://arxiv.org/pdf/2410.16155v2)

**Authors**: Tianyi Men, Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: With the development of large language models, they are widely used as agents in various fields. A key component of agents is memory, which stores vital information but is susceptible to jailbreak attacks. Existing research mainly focuses on single-agent attacks and shared memory attacks. However, real-world scenarios often involve independent memory. In this paper, we propose the Troublemaker Makes Chaos in Honest Town (TMCHT) task, a large-scale, multi-agent, multi-topology text-based attack evaluation framework. TMCHT involves one attacker agent attempting to mislead an entire society of agents. We identify two major challenges in multi-agent attacks: (1) Non-complete graph structure, (2) Large-scale systems. We attribute these challenges to a phenomenon we term toxicity disappearing. To address these issues, we propose an Adversarial Replication Contagious Jailbreak (ARCJ) method, which optimizes the retrieval suffix to make poisoned samples more easily retrieved and optimizes the replication suffix to make poisoned samples have contagious ability. We demonstrate the superiority of our approach in TMCHT, with 23.51%, 18.95%, and 52.93% improvements in line topology, star topology, and 100-agent settings. Encourage community attention to the security of multi-agent systems.

摘要: 随着大型语言模型的发展，它们被广泛用作各个领域的代理。代理的一个关键组成部分是内存，它存储重要信息，但容易受到越狱攻击。现有的研究主要集中在单代理攻击和共享内存攻击上。然而，现实世界的场景通常涉及独立记忆。在本文中，我们提出了Troubblemaker Makes Chaos in Honest Town（TMCHT）任务，这是一个大规模、多代理、多基于文本的攻击评估框架。TMCHT涉及一名攻击者特工试图误导整个特工社会。我们确定了多代理攻击中的两个主要挑战：（1）不完整的图结构，（2）大规模系统。我们将这些挑战归因于我们称之为毒性消失的现象。为了解决这些问题，我们提出了一种对抗复制传染越狱（ASCJ）方法，该方法优化检索后缀以使中毒样本更容易检索，并优化复制后缀以使中毒样本具有传染能力。我们在TMCHT中证明了我们的方法的优越性，线路布局、星型布局和100个代理设置分别提高了23.51%、18.95%和52.93%。鼓励社区关注多代理系统的安全性。



## **43. In-context learning for the classification of manipulation techniques in phishing emails**

用于网络钓鱼电子邮件操纵技术分类的上下文学习 cs.CR

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.22515v1) [paper-pdf](http://arxiv.org/pdf/2506.22515v1)

**Authors**: Antony Dalmiere, Guillaume Auriol, Vincent Nicomette, Pascal Marchand

**Abstract**: Traditional phishing detection often overlooks psychological manipulation. This study investigates using Large Language Model (LLM) In-Context Learning (ICL) for fine-grained classification of phishing emails based on a taxonomy of 40 manipulation techniques. Using few-shot examples with GPT-4o-mini on real-world French phishing emails (SignalSpam), we evaluated performance against a human-annotated test set (100 emails). The approach effectively identifies prevalent techniques (e.g., Baiting, Curiosity Appeal, Request For Minor Favor) with a promising accuracy of 0.76. This work demonstrates ICL's potential for nuanced phishing analysis and provides insights into attacker strategies.

摘要: 传统的网络钓鱼检测常常忽视心理操纵。本研究调查了使用大型语言模型（LLM）内上下文学习（ICL）基于40种操纵技术的分类法对网络钓鱼电子邮件进行细粒度分类。使用GPT-4 o-mini对现实世界的法语网络钓鱼电子邮件（SignalSpam）进行了几个示例，我们针对人类注释的测试集（100封电子邮件）评估了性能。该方法有效地识别流行技术（例如，诱饵、好奇心呼吁、请求小恩惠），准确率为0.76。这项工作展示了ICL进行细致入微的网络钓鱼分析的潜力，并提供了对攻击者策略的见解。



## **44. Doppelganger Method: Breaking Role Consistency in LLM Agent via Prompt-based Transferable Adversarial Attack**

分身方法：通过基于预算的可转移对抗攻击打破LLM代理中的角色一致性 cs.AI

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.14539v2) [paper-pdf](http://arxiv.org/pdf/2506.14539v2)

**Authors**: Daewon Kang, YeongHwan Shin, Doyeon Kim, Kyu-Hwan Jung, Meong Hi Son

**Abstract**: Since the advent of large language models, prompt engineering now enables the rapid, low-effort creation of diverse autonomous agents that are already in widespread use. Yet this convenience raises urgent concerns about the safety, robustness, and behavioral consistency of the underlying prompts, along with the pressing challenge of preventing those prompts from being exposed to user's attempts. In this paper, we propose the ''Doppelganger method'' to demonstrate the risk of an agent being hijacked, thereby exposing system instructions and internal information. Next, we define the ''Prompt Alignment Collapse under Adversarial Transfer (PACAT)'' level to evaluate the vulnerability to this adversarial transfer attack. We also propose a ''Caution for Adversarial Transfer (CAT)'' prompt to counter the Doppelganger method. The experimental results demonstrate that the Doppelganger method can compromise the agent's consistency and expose its internal information. In contrast, CAT prompts enable effective defense against this adversarial attack.

摘要: 自从大型语言模型的出现以来，即时工程现在可以快速、低努力地创建已经广泛使用的各种自治代理。然而，这种便利性引发了人们对底层提示的安全性、稳健性和行为一致性的紧迫担忧，以及防止这些提示暴露于用户尝试的紧迫挑战。在本文中，我们提出了“Doppelganger方法”来演示代理被劫持从而暴露系统指令和内部信息的风险。接下来，我们定义“对抗性转移下的提示对齐崩溃（PACAT RST）”级别来评估这种对抗性转移攻击的脆弱性。我们还提出了“对抗性转移的警告（CAT）”提示来对抗Doppelganger方法。实验结果表明，Doppelganger方法会损害代理的一致性并暴露其内部信息。相比之下，CAT提示可以有效防御这种对抗性攻击。



## **45. WiS Platform: Enhancing Evaluation of LLM-Based Multi-Agent Systems Through Game-Based Analysis**

WiS平台：通过基于游戏的分析增强对基于LLM的多智能体系统的评估 cs.AI

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2412.03359v2) [paper-pdf](http://arxiv.org/pdf/2412.03359v2)

**Authors**: Chengwei Hu, Jianhui Zheng, Yancheng He, Hangyu Guo, Junguang Jiang, Han Zhu, Kai Sun, Yuning Jiang, Wenbo Su, Bo Zheng

**Abstract**: Recent advancements in autonomous multi-agent systems (MAS) based on large language models (LLMs) have enhanced the application scenarios and improved the capability of LLMs to handle complex tasks. Despite demonstrating effectiveness, existing studies still evidently struggle to evaluate, analysis, and reproducibility of LLM-based MAS. In this paper, to facilitate the research on LLM-based MAS, we introduce an open, scalable, and real-time updated platform for accessing and analyzing the LLM-based MAS based on the games Who is Spy?" (WiS). Our platform is featured with three main worths: (1) a unified model evaluate interface that supports models available on Hugging Face; (2) real-time updated leaderboard for model evaluation; (3) a comprehensive evaluation covering game-winning rates, attacking, defense strategies, and reasoning of LLMs. To rigorously test WiS, we conduct extensive experiments coverage of various open- and closed-source LLMs, we find that different agents exhibit distinct and intriguing behaviors in the game. The experimental results demonstrate the effectiveness and efficiency of our platform in evaluating LLM-based MAS. Our platform and its documentation are publicly available at https://whoisspy.ai/.

摘要: 基于大型语言模型（LLM）的自治多智能体系统（MAS）的最新进展增强了应用场景并提高了LLM处理复杂任务的能力。尽管证明了有效性，但现有的研究显然仍然难以评估、分析基于LLM的MAS的可重复性。本文为了促进基于LLM的MAS的研究，我们引入了一个开放的、可扩展的、实时更新的平台，用于访问和分析基于LLM的MAS基于游戏《谁是间谍？》”（WiS）。我们的平台具有三个主要功能：（1）统一的模型评估界面，支持Hugging Face上可用的模型;（2）实时更新的模型评估排行榜;（3）涵盖游戏获胜率、攻击、防御策略和LLM推理的全面评估。为了严格测试WiS，我们对各种开源和开源LLM进行了广泛的实验，我们发现不同的代理在游戏中表现出不同且有趣的行为。实验结果证明了我们的平台在评估基于LLM的MAS方面的有效性和效率。我们的平台及其文档可在https://whoisspy.ai/上公开获取。



## **46. AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text**

AgentStealth：加强大型语言模型以简化用户生成的文本 cs.CL

This work has been submitted to NeurIPS 2025. Under review

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.22508v1) [paper-pdf](http://arxiv.org/pdf/2506.22508v1)

**Authors**: Chenyang Shao, Tianxing Li, Chenhao Pu, Fengli Xu, Yong Li

**Abstract**: In today's digital world, casual user-generated content often contains subtle cues that may inadvertently expose sensitive personal attributes. Such risks underscore the growing importance of effective text anonymization to safeguard individual privacy. However, existing methods either rely on rigid replacements that damage utility or cloud-based LLMs that are costly and pose privacy risks. To address these issues, we explore the use of locally deployed smaller-scale language models (SLMs) for anonymization. Yet training effective SLMs remains challenging due to limited high-quality supervision. To address the challenge, we propose AgentStealth, a self-reinforcing LLM anonymization framework.First, we introduce an adversarial anonymization workflow enhanced by In-context Contrastive Learning and Adaptive Utility-Aware Control. Second, we perform supervised adaptation of SLMs using high-quality data collected from the workflow, which includes both anonymization and attack signals. Finally, we apply online reinforcement learning where the model leverages its internal adversarial feedback to iteratively improve anonymization performance. Experiments on two datasets show that our method outperforms baselines in both anonymization effectiveness (+12.3%) and utility (+6.8%). Our lightweight design supports direct deployment on edge devices, avoiding cloud reliance and communication-based privacy risks. Our code is open-source at https://github.com/tsinghua-fib-lab/AgentStealth.

摘要: 在当今的数字世界中，用户生成的随意内容通常包含微妙的线索，这些线索可能会无意中暴露敏感的个人属性。此类风险凸显了有效的文本匿名化对于保护个人隐私的重要性。然而，现有方法要么依赖于损害公用事业的严格替代品，要么依赖于成本高昂并构成隐私风险的基于云的LLM。为了解决这些问题，我们探索使用本地部署的较小规模语言模型（SLC）进行匿名化。然而，由于高质量的监督有限，培训有效的CRM仍然具有挑战性。为了应对这一挑战，我们提出了AgentStealth，这是一个自我增强的LLM匿名化框架。首先，我们引入了一种由上下文内对比学习和自适应实用性感知控制增强的对抗性匿名化工作流程。其次，我们使用从工作流程中收集的高质量数据（包括匿名化和攻击信号）对CRM进行监督调整。最后，我们应用在线强化学习，其中模型利用其内部对抗反馈来迭代改进匿名化性能。对两个数据集的实验表明，我们的方法在匿名有效性（+12.3%）和实用性（+6.8%）方面都优于基线。我们的轻量级设计支持在边缘设备上直接部署，避免云依赖和基于通信的隐私风险。我们的代码是开源的，网址是https://github.com/tsinghua-fib-lab/AgentStealth。



## **47. E-FreeM2: Efficient Training-Free Multi-Scale and Cross-Modal News Verification via MLLMs**

E-FreeM 2：通过MLLM进行高效的免培训多规模和跨模式新闻验证 cs.MM

Accepted to AsiaCCS 2025 @ SCID

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.20944v1) [paper-pdf](http://arxiv.org/pdf/2506.20944v1)

**Authors**: Van-Hoang Phan, Long-Khanh Pham, Dang Vu, Anh-Duy Tran, Minh-Son Dao

**Abstract**: The rapid spread of misinformation in mobile and wireless networks presents critical security challenges. This study introduces a training-free, retrieval-based multimodal fact verification system that leverages pretrained vision-language models and large language models for credibility assessment. By dynamically retrieving and cross-referencing trusted data sources, our approach mitigates vulnerabilities of traditional training-based models, such as adversarial attacks and data poisoning. Additionally, its lightweight design enables seamless edge device integration without extensive on-device processing. Experiments on two fact-checking benchmarks achieve SOTA results, confirming its effectiveness in misinformation detection and its robustness against various attack vectors, highlighting its potential to enhance security in mobile and wireless communication environments.

摘要: 移动和无线网络中错误信息的迅速传播带来了严峻的安全挑战。本研究引入了一种免训练、基于检索的多模式事实验证系统，该系统利用预先训练的视觉语言模型和大型语言模型进行可信度评估。通过动态检索和交叉引用可信数据源，我们的方法减轻了传统基于训练的模型的漏洞，例如对抗性攻击和数据中毒。此外，其轻量级设计可以实现无缝边缘设备集成，无需进行大量的设备上处理。在两个事实检查基准上进行的实验获得了SOTA结果，证实了其在错误信息检测方面的有效性及其对各种攻击载体的鲁棒性，凸显了其增强移动和无线通信环境安全性的潜力。



## **48. SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning**

SABRE-FL：针对联邦即时学习的选择性、准确的后门拒绝 cs.CR

**SubmitDate**: 2025-06-25    [abs](http://arxiv.org/abs/2506.22506v1) [paper-pdf](http://arxiv.org/pdf/2506.22506v1)

**Authors**: Momin Ahmad Khan, Yasra Chandio, Fatima Muhammad Anwar

**Abstract**: Federated Prompt Learning has emerged as a communication-efficient and privacy-preserving paradigm for adapting large vision-language models like CLIP across decentralized clients. However, the security implications of this setup remain underexplored. In this work, we present the first study of backdoor attacks in Federated Prompt Learning. We show that when malicious clients inject visually imperceptible, learnable noise triggers into input images, the global prompt learner becomes vulnerable to targeted misclassification while still maintaining high accuracy on clean inputs. Motivated by this vulnerability, we propose SABRE-FL, a lightweight, modular defense that filters poisoned prompt updates using an embedding-space anomaly detector trained offline on out-of-distribution data. SABRE-FL requires no access to raw client data or labels and generalizes across diverse datasets. We show, both theoretically and empirically, that malicious clients can be reliably identified and filtered using an embedding-based detector. Across five diverse datasets and four baseline defenses, SABRE-FL outperforms all baselines by significantly reducing backdoor accuracy while preserving clean accuracy, demonstrating strong empirical performance and underscoring the need for robust prompt learning in future federated systems.

摘要: 联合提示学习已成为一种高效通信和保护隐私的范式，用于在去中心化客户端之间适应CLIP等大型视觉语言模型。然而，这种设置的安全影响仍然没有得到充分的研究。在这项工作中，我们首次介绍了联邦提示学习中后门攻击的研究。我们表明，当恶意客户端将视觉上不可感知的、可学习的噪音触发器注入输入图像时，全局提示学习器变得容易受到有针对性的错误分类的影响，同时仍然保持干净输入的高准确性。受此漏洞的启发，我们提出了SABRE-FL，这是一种轻量级的模块化防御，它使用对非分布数据进行离线训练的嵌入空间异常检测器来过滤有毒的提示更新。SABRE-FL不需要访问原始客户数据或标签，并在不同的数据集中进行概括。我们从理论上和经验上表明，可以使用基于嵌入的检测器可靠地识别和过滤恶意客户端。在五个不同的数据集和四种基线防御中，SABRE-FL通过显着降低后门准确性同时保持清晰的准确性来优于所有基线，展示了强大的经验性能并强调了未来联邦系统对稳健即时学习的需求。



## **49. Leaner Training, Lower Leakage: Revisiting Memorization in LLM Fine-Tuning with LoRA**

更精简的训练，更低的泄漏：重新审视LLM微调与LoRA cs.LG

**SubmitDate**: 2025-06-25    [abs](http://arxiv.org/abs/2506.20856v1) [paper-pdf](http://arxiv.org/pdf/2506.20856v1)

**Authors**: Fei Wang, Baochun Li

**Abstract**: Memorization in large language models (LLMs) makes them vulnerable to data extraction attacks. While pre-training memorization has been extensively studied, fewer works have explored its impact in fine-tuning, particularly for LoRA fine-tuning, a widely adopted parameter-efficient method.   In this work, we re-examine memorization in fine-tuning and uncover a surprising divergence from prior findings across different fine-tuning strategies. Factors such as model scale and data duplication, which strongly influence memorization in pre-training and full fine-tuning, do not follow the same trend in LoRA fine-tuning. Using a more relaxed similarity-based memorization metric, we demonstrate that LoRA significantly reduces memorization risks compared to full fine-tuning, while still maintaining strong task performance.

摘要: 大型语言模型（LLM）中的并行化使它们容易受到数据提取攻击。虽然训练前记忆已经得到了广泛的研究，但很少有作品探索其对微调的影响，特别是对于LoRA微调（一种广泛采用的参数高效方法）。   在这项工作中，我们重新检查了微调中的记忆，并发现了不同微调策略与先前发现的令人惊讶的差异。模型规模和数据重复等因素强烈影响预训练和完全微调中的记忆，但在LoRA微调中并不遵循相同的趋势。使用更宽松的基于相似性的记忆指标，我们证明与完全微调相比，LoRA显着降低了记忆风险，同时仍然保持强劲的任务性能。



## **50. Poster: Enhancing GNN Robustness for Network Intrusion Detection via Agent-based Analysis**

海报：通过基于代理的分析增强GNN网络入侵检测的鲁棒性 cs.CR

Poster accepted at the 10th IEEE European Symposium on Security and  Privacy (Euro S&P 2025)

**SubmitDate**: 2025-06-25    [abs](http://arxiv.org/abs/2506.20806v1) [paper-pdf](http://arxiv.org/pdf/2506.20806v1)

**Authors**: Zhonghao Zhan, Huichi Zhou, Hamed Haddadi

**Abstract**: Graph Neural Networks (GNNs) show great promise for Network Intrusion Detection Systems (NIDS), particularly in IoT environments, but suffer performance degradation due to distribution drift and lack robustness against realistic adversarial attacks. Current robustness evaluations often rely on unrealistic synthetic perturbations and lack demonstrations on systematic analysis of different kinds of adversarial attack, which encompass both black-box and white-box scenarios. This work proposes a novel approach to enhance GNN robustness and generalization by employing Large Language Models (LLMs) in an agentic pipeline as simulated cybersecurity expert agents. These agents scrutinize graph structures derived from network flow data, identifying and potentially mitigating suspicious or adversarially perturbed elements before GNN processing. Our experiments, using a framework designed for realistic evaluation and testing with a variety of adversarial attacks including a dataset collected from physical testbed experiments, demonstrate that integrating LLM analysis can significantly improve the resilience of GNN-based NIDS against challenges, showcasing the potential of LLM agent as a complementary layer in intrusion detection architectures.

摘要: 图形神经网络（GNN）在网络入侵检测系统（NIDS）方面表现出了巨大的前景，特别是在物联网环境中，但由于分布漂移而导致性能下降，并且缺乏针对现实对抗攻击的鲁棒性。当前的稳健性评估通常依赖于不切实际的合成扰动，并且缺乏对不同类型对抗攻击（包括黑匣子和白盒场景）的系统分析的演示。这项工作提出了一种新颖的方法，通过在代理管道中使用大型语言模型（LLM）作为模拟的网络安全专家代理来增强GNN的鲁棒性和概括性。这些代理检查从网络流数据派生的图结构，在GNN处理之前识别并可能减轻可疑或敌对干扰的元素。我们的实验使用了一个框架，该框架设计用于对各种对抗性攻击进行现实评估和测试，包括从物理测试台实验中收集的数据集，证明了集成LLM分析可以显着提高基于GNN的NIDS对挑战的弹性，展示了LLM代理作为入侵检测体系结构中补充层的潜力。



