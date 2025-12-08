# Latest Large Language Model Attack Papers
**update at 2025-12-08 15:35:13**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. VRSA: Jailbreaking Multimodal Large Language Models through Visual Reasoning Sequential Attack**

VRSA：通过视觉推理序列攻击破解多模式大型语言模型 cs.CV

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05853v1) [paper-pdf](https://arxiv.org/pdf/2512.05853v1)

**Authors**: Shiji Zhao, Shukun Xiong, Yao Huang, Yan Jin, Zhenyu Wu, Jiyang Guan, Ranjie Duan, Jialing Tao, Hui Xue, Xingxing Wei

**Abstract**: Multimodal Large Language Models (MLLMs) are widely used in various fields due to their powerful cross-modal comprehension and generation capabilities. However, more modalities bring more vulnerabilities to being utilized for jailbreak attacks, which induces MLLMs to output harmful content. Due to the strong reasoning ability of MLLMs, previous jailbreak attacks try to explore reasoning safety risk in text modal, while similar threats have been largely overlooked in the visual modal. To fully evaluate potential safety risks in the visual reasoning task, we propose Visual Reasoning Sequential Attack (VRSA), which induces MLLMs to gradually externalize and aggregate complete harmful intent by decomposing the original harmful text into several sequentially related sub-images. In particular, to enhance the rationality of the scene in the image sequence, we propose Adaptive Scene Refinement to optimize the scene most relevant to the original harmful query. To ensure the semantic continuity of the generated image, we propose Semantic Coherent Completion to iteratively rewrite each sub-text combined with contextual information in this scene. In addition, we propose Text-Image Consistency Alignment to keep the semantical consistency. A series of experiments demonstrates that the VRSA can achieve a higher attack success rate compared with the state-of-the-art jailbreak attack methods on both the open-source and closed-source MLLMs such as GPT-4o and Claude-4.5-Sonnet.

摘要: 多模式大型语言模型（MLLM）因其强大的跨模式理解和生成能力而被广泛应用于各个领域。然而，更多的模式会带来更多被用于越狱攻击的漏洞，从而导致MLLM输出有害内容。由于MLLM推理能力强，之前的越狱攻击试图在文本模式中探索推理安全风险，而类似的威胁在视觉模式中基本上被忽视。为了充分评估视觉推理任务中潜在的安全风险，我们提出了视觉推理序列攻击（VRSA），通过将原始有害文本分解为几个顺序相关的子图像，诱导MLLM逐渐外部化和聚合完整的有害意图。特别是，为了增强图像序列中场景的合理性，我们提出自适应场景细化来优化与原始有害查询最相关的场景。为了确保生成图像的语义连续性，我们提出了语义连贯完成来迭代重写该场景中结合上下文信息的每个子文本。此外，我们还提出了文本-图像一致性对齐来保持语义一致性。一系列实验表明，与最先进的越狱攻击方法相比，VRSA可以在GPT-4 o和Claude-4.5-Sonnet等开源和闭源MLLM上实现更高的攻击成功率。



## **2. ARGUS: Defending Against Multimodal Indirect Prompt Injection via Steering Instruction-Following Behavior**

ARGucci：通过转向指令遵循行为防御多模式间接提示注射 cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05745v1) [paper-pdf](https://arxiv.org/pdf/2512.05745v1)

**Authors**: Weikai Lu, Ziqian Zeng, Kehua Zhang, Haoran Li, Huiping Zhuang, Ruidong Wang, Cen Chen, Hao Peng

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly vulnerable to multimodal Indirect Prompt Injection (IPI) attacks, which embed malicious instructions in images, videos, or audio to hijack model behavior. Existing defenses, designed primarily for text-only LLMs, are unsuitable for countering these multimodal threats, as they are easily bypassed, modality-dependent, or generalize poorly. Inspired by activation steering researches, we hypothesize that a robust, general defense independent of modality can be achieved by steering the model's behavior in the representation space. Through extensive experiments, we discover that the instruction-following behavior of MLLMs is encoded in a subspace. Steering along directions within this subspace can enforce adherence to user instructions, forming the basis of a defense. However, we also found that a naive defense direction could be coupled with a utility-degrading direction, and excessive intervention strength harms model performance. To address this, we propose ARGUS, which searches for an optimal defense direction within the safety subspace that decouples from the utility degradation direction, further combining adaptive strength steering to achieve a better safety-utility trade-off. ARGUS also introduces lightweight injection detection stage to activate the defense on-demand, and a post-filtering stage to verify defense success. Experimental results show that ARGUS can achieve robust defense against multimodal IPI while maximally preserving the MLLM's utility.

摘要: 多模式大型语言模型（MLLM）越来越容易受到多模式间接提示注入（IPI）攻击的影响，这些攻击将恶意指令嵌入图像、视频或音频中以劫持模型行为。现有的防御主要为纯文本的LLM设计，不适合对抗这些多模式威胁，因为它们很容易被绕过、依赖于模式或概括性较差。受激活引导研究的启发，我们假设可以通过引导模型在表示空间中的行为来实现独立于形态的稳健、通用防御。通过大量实验，我们发现MLLM的描述跟随行为被编码在子空间中。沿着该子空间内的方向转向可以强制遵守用户指令，从而形成防御的基础。然而，我们还发现，天真的防御方向可能会与效用下降的方向相结合，过度的干预强度会损害模型性能。为了解决这个问题，我们提出了ARGucci，它在安全子空间内搜索与效用降级方向并行的最佳防御方向，进一步结合自适应强度引导以实现更好的安全-效用权衡。ARGucci还引入了轻量级注入检测阶段以按需激活防御，以及后过滤阶段以验证防御成功。实验结果表明，ARGUS可以实现强大的防御多模态IPI，同时最大限度地保持MLLM的效用。



## **3. Matching Ranks Over Probability Yields Truly Deep Safety Alignment**

超越概率的匹配排名产生真正深入的安全一致 cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05518v1) [paper-pdf](https://arxiv.org/pdf/2512.05518v1)

**Authors**: Jason Vega, Gagandeep Singh

**Abstract**: A frustratingly easy technique known as the prefilling attack has been shown to effectively circumvent the safety alignment of frontier LLMs by simply prefilling the assistant response with an affirmative prefix before decoding. In response, recent work proposed a supervised fine-tuning (SFT) defense using data augmentation to achieve a \enquote{deep} safety alignment, allowing the model to generate natural language refusals immediately following harmful prefills. Unfortunately, we show in this work that the "deep" safety alignment produced by such an approach is in fact not very deep. A generalization of the prefilling attack, which we refer to as the Rank-Assisted Prefilling (RAP) attack, can effectively extract harmful content from models fine-tuned with the data augmentation defense by selecting low-probability "harmful" tokens from the top 20 predicted next tokens at each step (thus ignoring high-probability "refusal" tokens). We argue that this vulnerability is enabled due to the "gaming" of the SFT objective when the target distribution entropies are low, where low fine-tuning loss is achieved by shifting large probability mass to a small number of refusal tokens while neglecting the high ranks of harmful tokens. We then propose a new perspective on achieving deep safety alignment by matching the token ranks of the target distribution, rather than their probabilities. This perspective yields a surprisingly simple fix to the data augmentation defense based on regularizing the attention placed on harmful prefill tokens, an approach we call PRefill attEntion STOpping (PRESTO). Adding PRESTO yields up to a 4.7x improvement in the mean StrongREJECT score under RAP attacks across three popular open-source LLMs, with low impact to model utility.

摘要: 一种名为预填充攻击的简单技术已被证明可以通过在解码之前简单地预填充辅助响应来有效规避前沿LLM的安全对齐。作为回应，最近的工作提出了一种监督式微调（SFT）防御，使用数据增强来实现安全对齐，允许模型在有害预填充后立即生成自然语言拒绝。不幸的是，我们在这项工作中表明，这种方法产生的“深度”安全调整实际上并不很深。预填充攻击的概括（我们称之为排名辅助预填充（RAP）攻击）可以通过从每一步预测的前20个下一个令牌中选择低概率的“有害”令牌，从用数据增强防御微调的模型中有效地提取有害内容（从而忽略高概率的“拒绝”令牌）。我们认为，这种漏洞是启用由于“游戏”的SFT目标时，目标分布熵低，其中低微调损失是通过转移大概率质量的拒绝令牌的数量少，而忽略了高排名的有害令牌。然后，我们提出了一个新的视角，通过匹配目标分布的令牌等级，而不是它们的概率，来实现深度安全对齐。这种观点产生了一个令人惊讶的简单的修复数据增强防御的基础上正规化的注意力放在有害的预填充令牌，一种方法，我们称之为预填充attEntion停止（PRESTO）。在三种流行的开源LLM中，在RAP攻击下，添加PRESTO可以使平均StrongRESISTANCE得分提高4.7倍，对模型实用性的影响很小。



## **4. TeleAI-Safety: A comprehensive LLM jailbreaking benchmark towards attacks, defenses, and evaluations**

TeleAI-Safety：针对攻击、防御和评估的全面LLM越狱基准 cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05485v1) [paper-pdf](https://arxiv.org/pdf/2512.05485v1)

**Authors**: Xiuyuan Chen, Jian Zhao, Yuxiang He, Yuan Xun, Xinwei Liu, Yanshu Li, Huilin Zhou, Wei Cai, Ziyan Shi, Yuchen Yuan, Tianle Zhang, Chi Zhang, Xuelong Li

**Abstract**: While the deployment of large language models (LLMs) in high-value industries continues to expand, the systematic assessment of their safety against jailbreak and prompt-based attacks remains insufficient. Existing safety evaluation benchmarks and frameworks are often limited by an imbalanced integration of core components (attack, defense, and evaluation methods) and an isolation between flexible evaluation frameworks and standardized benchmarking capabilities. These limitations hinder reliable cross-study comparisons and create unnecessary overhead for comprehensive risk assessment. To address these gaps, we present TeleAI-Safety, a modular and reproducible framework coupled with a systematic benchmark for rigorous LLM safety evaluation. Our framework integrates a broad collection of 19 attack methods (including one self-developed method), 29 defense methods, and 19 evaluation methods (including one self-developed method). With a curated attack corpus of 342 samples spanning 12 distinct risk categories, the TeleAI-Safety benchmark conducts extensive evaluations across 14 target models. The results reveal systematic vulnerabilities and model-specific failure cases, highlighting critical trade-offs between safety and utility, and identifying potential defense patterns for future optimization. In practical scenarios, TeleAI-Safety can be flexibly adjusted with customized attack, defense, and evaluation combinations to meet specific demands. We release our complete code and evaluation results to facilitate reproducible research and establish unified safety baselines.

摘要: 尽管大型语言模型（LLM）在高价值行业的部署持续扩大，但对其针对越狱和预算攻击的安全性的系统评估仍然不足。现有的安全评估基准和框架通常受到核心组件（攻击、防御和评估方法）的不平衡集成以及灵活评估框架和标准化基准能力之间的隔离的限制。这些限制阻碍了可靠的交叉研究比较，并为全面风险评估带来了不必要的费用。为了解决这些差距，我们提出了TeleAI-Safety，这是一个模块化、可重复的框架，结合了严格LLM安全评估的系统基准。我们的框架集成了19种攻击方法（包括一种自主开发的方法），29种防御方法和19种评估方法（包括一种自主开发的方法）。TeleAI-Safety基准测试使用了涵盖12个不同风险类别的342个样本的策划攻击语料库，对14个目标模型进行了广泛的评估。结果揭示了系统漏洞和特定于模型的故障案例，突出了安全性和实用性之间的关键权衡，并确定了未来优化的潜在防御模式。在实际场景中，TeleAI-Safety可以灵活调整自定义攻击、防御和评估组合，以满足特定需求。我们发布完整的代码和评估结果，以促进可重复的研究并建立统一的安全基线。



## **5. Exposing Pink Slime Journalism: Linguistic Signatures and Robust Detection Against LLM-Generated Threats**

揭露Pink Slime新闻：语言签名和针对LLM生成的威胁的稳健检测 cs.CL

Published in RANLP 2025

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05331v1) [paper-pdf](https://arxiv.org/pdf/2512.05331v1)

**Authors**: Sadat Shahriar, Navid Ayoobi, Arjun Mukherjee, Mostafa Musharrat, Sai Vishnu Vamsi

**Abstract**: The local news landscape, a vital source of reliable information for 28 million Americans, faces a growing threat from Pink Slime Journalism, a low-quality, auto-generated articles that mimic legitimate local reporting. Detecting these deceptive articles requires a fine-grained analysis of their linguistic, stylistic, and lexical characteristics. In this work, we conduct a comprehensive study to uncover the distinguishing patterns of Pink Slime content and propose detection strategies based on these insights. Beyond traditional generation methods, we highlight a new adversarial vector: modifications through large language models (LLMs). Our findings reveal that even consumer-accessible LLMs can significantly undermine existing detection systems, reducing their performance by up to 40% in F1-score. To counter this threat, we introduce a robust learning framework specifically designed to resist LLM-based adversarial attacks and adapt to the evolving landscape of automated pink slime journalism, and showed and improvement by up to 27%.

摘要: 当地新闻格局是2800万美国人可靠信息的重要来源，但面临着来自Pink Slime Journalism的日益增长的威胁，Pink Slime Journalism是一种模仿合法当地报道的低质量自动生成文章。检测这些欺骗性文章需要对其语言、文体和词汇特征进行细粒度分析。在这项工作中，我们进行了一项全面的研究，以揭示Pink Slime内容的区别模式，并根据这些见解提出检测策略。除了传统的生成方法之外，我们还强调了一种新的对抗性载体：通过大型语言模型（LLM）进行修改。我们的研究结果表明，即使是消费者可访问的LLM也会显着破坏现有的检测系统，使其F1评分的性能降低高达40%。为了应对这一威胁，我们引入了一个强大的学习框架，专门设计用于抵抗基于LLM的对抗攻击并适应自动化粉红粘液新闻不断变化的格局，并表现出高达27%的改进。



## **6. Chameleon: Adaptive Adversarial Agents for Scaling-Based Visual Prompt Injection in Multimodal AI Systems**

Chameleon：用于多模式人工智能系统中基于扩展的视觉提示注入的自适应对抗代理 cs.AI

5 pages, 2 figures, IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04895v1) [paper-pdf](https://arxiv.org/pdf/2512.04895v1)

**Authors**: M Zeeshan, Saud Satti

**Abstract**: Multimodal Artificial Intelligence (AI) systems, particularly Vision-Language Models (VLMs), have become integral to critical applications ranging from autonomous decision-making to automated document processing. As these systems scale, they rely heavily on preprocessing pipelines to handle diverse inputs efficiently. However, this dependency on standard preprocessing operations, specifically image downscaling, creates a significant yet often overlooked security vulnerability. While intended for computational optimization, scaling algorithms can be exploited to conceal malicious visual prompts that are invisible to human observers but become active semantic instructions once processed by the model. Current adversarial strategies remain largely static, failing to account for the dynamic nature of modern agentic workflows. To address this gap, we propose Chameleon, a novel, adaptive adversarial framework designed to expose and exploit scaling vulnerabilities in production VLMs. Unlike traditional static attacks, Chameleon employs an iterative, agent-based optimization mechanism that dynamically refines image perturbations based on the target model's real-time feedback. This allows the framework to craft highly robust adversarial examples that survive standard downscaling operations to hijack downstream execution. We evaluate Chameleon against Gemini 2.5 Flash model. Our experiments demonstrate that Chameleon achieves an Attack Success Rate (ASR) of 84.5% across varying scaling factors, significantly outperforming static baseline attacks which average only 32.1%. Furthermore, we show that these attacks effectively compromise agentic pipelines, reducing decision-making accuracy by over 45% in multi-step tasks. Finally, we discuss the implications of these vulnerabilities and propose multi-scale consistency checks as a necessary defense mechanism.

摘要: 多模式人工智能（AI）系统，特别是视觉语言模型（VLM），已成为从自主决策到自动文档处理等关键应用不可或缺的一部分。随着这些系统的扩展，它们严重依赖预处理管道来有效处理不同的输入。然而，这种对标准预处理操作（特别是图像缩减）的依赖会产生一个严重但经常被忽视的安全漏洞。虽然缩放算法旨在实现计算优化，但可以利用缩放算法来隐藏恶意视觉提示，这些提示对人类观察者来说是不可见的，但一旦被模型处理就变成了活动的语义指令。当前的对抗策略在很大程度上仍然是静态的，未能考虑到现代代理工作流程的动态性质。为了解决这一差距，我们提出了Chameleon，这是一种新颖的、自适应的对抗框架，旨在暴露和利用生产VLM中的扩展漏洞。与传统的静态攻击不同，Chameleon采用基于代理的迭代优化机制，该机制根据目标模型的实时反馈动态细化图像扰动。这使得该框架能够制作高度稳健的对抗性示例，这些示例能够经受住标准缩减操作以劫持下游执行的考验。我们根据Gemini 2.5 Flash模型评估Chameleon。我们的实验表明，Chameleon在不同的缩放因子下实现了84.5%的攻击成功率（ASB），显着优于平均仅为32.1%的静态基线攻击。此外，我们表明这些攻击有效地损害了代理管道，使多步骤任务中的决策准确性降低了45%以上。最后，我们讨论了这些漏洞的影响，并提出多规模一致性检查作为必要的防御机制。



## **7. SoK: a Comprehensive Causality Analysis Framework for Large Language Model Security**

SoK：大型语言模型安全性的全面因果分析框架 cs.CR

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04841v1) [paper-pdf](https://arxiv.org/pdf/2512.04841v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but remain vulnerable to adversarial manipulations such as jailbreaking, where crafted prompts bypass safety mechanisms. Understanding the causal factors behind such vulnerabilities is essential for building reliable defenses.   In this work, we introduce a unified causality analysis framework that systematically supports all levels of causal investigation in LLMs, ranging from token-level, neuron-level, and layer-level interventions to representation-level analysis. The framework enables consistent experimentation and comparison across diverse causality-based attack and defense methods. Accompanying this implementation, we provide the first comprehensive survey of causality-driven jailbreak studies and empirically evaluate the framework on multiple open-weight models and safety-critical benchmarks including jailbreaks, hallucination detection, backdoor identification, and fairness evaluation. Our results reveal that: (1) targeted interventions on causally critical components can reliably modify safety behavior; (2) safety-related mechanisms are highly localized (i.e., concentrated in early-to-middle layers with only 1--2\% of neurons exhibiting causal influence); and (3) causal features extracted from our framework achieve over 95\% detection accuracy across multiple threat types.   By bridging theoretical causality analysis and practical model safety, our framework establishes a reproducible foundation for research on causality-based attacks, interpretability, and robust attack detection and mitigation in LLMs. Code is available at https://github.com/Amadeuszhao/SOK_Casuality.

摘要: 大型语言模型（LLM）表现出非凡的能力，但仍然容易受到越狱等敌对操纵的影响，其中精心设计的提示绕过了安全机制。了解此类漏洞背后的因果因素对于构建可靠的防御至关重要。   在这项工作中，我们引入了一个统一的因果关系分析框架，该框架系统地支持LLM中所有层面的因果关系调查，从代币层面、神经元层面和层层面干预到代表层面分析。该框架能够在各种基于偶然性的攻击和防御方法之间进行一致的实验和比较。伴随着这一实施，我们对疏忽驱动的越狱研究进行了首次全面调查，并对多个开放权重模型和安全关键基准（包括越狱、幻觉检测、后门识别和公平性评估）的框架进行了实证评估。我们的结果表明：（1）对因果关键部件进行有针对性的干预可以可靠地改变安全行为;（2）安全相关机制高度局部化（即，集中在早期到中层，只有1- 2%的神经元表现出因果影响）;以及（3）从我们的框架中提取的因果特征在多种威胁类型中实现了超过95%的检测准确率。   通过连接理论因果关系分析和实际模型安全性，我们的框架为LLM中基于因果关系的攻击、可解释性以及稳健的攻击检测和缓解的研究奠定了可重复的基础。代码可在https://github.com/Amadeuszhao/SOK_Casuality上获取。



## **8. ASTRIDE: A Security Threat Modeling Platform for Agentic-AI Applications**

ASTRIDE：用于统计人工智能应用的安全威胁建模平台 cs.AI

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04785v1) [paper-pdf](https://arxiv.org/pdf/2512.04785v1)

**Authors**: Eranga Bandara, Amin Hass, Ross Gore, Sachin Shetty, Ravi Mukkamala, Safdar H. Bouk, Xueping Liang, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan

**Abstract**: AI agent-based systems are becoming increasingly integral to modern software architectures, enabling autonomous decision-making, dynamic task execution, and multimodal interactions through large language models (LLMs). However, these systems introduce novel and evolving security challenges, including prompt injection attacks, context poisoning, model manipulation, and opaque agent-to-agent communication, that are not effectively captured by traditional threat modeling frameworks. In this paper, we introduce ASTRIDE, an automated threat modeling platform purpose-built for AI agent-based systems. ASTRIDE extends the classical STRIDE framework by introducing a new threat category, A for AI Agent-Specific Attacks, which encompasses emerging vulnerabilities such as prompt injection, unsafe tool invocation, and reasoning subversion, unique to agent-based applications. To automate threat modeling, ASTRIDE combines a consortium of fine-tuned vision-language models (VLMs) with the OpenAI-gpt-oss reasoning LLM to perform end-to-end analysis directly from visual agent architecture diagrams, such as data flow diagrams(DFDs). LLM agents orchestrate the end-to-end threat modeling automation process by coordinating interactions between the VLM consortium and the reasoning LLM. Our evaluations demonstrate that ASTRIDE provides accurate, scalable, and explainable threat modeling for next-generation intelligent systems. To the best of our knowledge, ASTRIDE is the first framework to both extend STRIDE with AI-specific threats and integrate fine-tuned VLMs with a reasoning LLM to fully automate diagram-driven threat modeling in AI agent-based applications.

摘要: 基于人工智能代理的系统越来越成为现代软件架构的组成部分，通过大型语言模型（LLM）实现自主决策、动态任务执行和多模式交互。然而，这些系统引入了新颖且不断发展的安全挑战，包括即时注入攻击、上下文中毒、模型操纵和不透明的代理到代理通信，传统威胁建模框架无法有效捕捉这些挑战。在本文中，我们介绍ASTRIDE，这是一个专门为基于人工智能代理的系统构建的自动化威胁建模平台。ASTRIDE通过引入新的威胁类别A（代表人工智能代理特定攻击）扩展了经典的WRIDE框架，其中包括基于代理的应用程序所独有的新漏洞，例如提示注入、不安全工具调用和推理颠覆。为了自动化威胁建模，ASTRIDE将微调视觉语言模型（VLM）联盟与OpenAI-gpt-oss推理LLM相结合，直接从视觉代理架构图（例如数据流图（DFD））执行端到端分析。LLM代理通过协调VLM联盟和推理LLM之间的交互来协调端到端威胁建模自动化流程。我们的评估表明，ASTRIDE为下一代智能系统提供了准确、可扩展且可解释的威胁建模。据我们所知，ASTRIDE是第一个既可以通过人工智能特定的威胁扩展WRIDE，又可以将微调的VLM与推理LLM集成，以在基于人工智能代理的应用程序中完全自动化任务驱动的威胁建模。



## **9. Automatic Attack Discovery for Few-Shot Class-Incremental Learning via Large Language Models**

通过大型语言模型实现少镜头类增量学习的自动攻击发现 cs.LG

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03882v1) [paper-pdf](https://arxiv.org/pdf/2512.03882v1)

**Authors**: Haidong Kang, Wei Wu, Hanling Wang

**Abstract**: Few-shot class incremental learning (FSCIL) is a more realistic and challenging paradigm in continual learning to incrementally learn unseen classes and overcome catastrophic forgetting on base classes with only a few training examples. Previous efforts have primarily centered around studying more effective FSCIL approaches. By contrast, less attention was devoted to thinking the security issues in contributing to FSCIL. This paper aims to provide a holistic study of the impact of attacks on FSCIL. We first derive insights by systematically exploring how human expert-designed attack methods (i.e., PGD, FGSM) affect FSCIL. We find that those methods either fail to attack base classes, or suffer from huge labor costs due to relying on huge expert knowledge. This highlights the need to craft a specialized attack method for FSCIL. Grounded in these insights, in this paper, we propose a simple yet effective ACraft method to automatically steer and discover optimal attack methods targeted at FSCIL by leveraging Large Language Models (LLMs) without human experts. Moreover, to improve the reasoning between LLMs and FSCIL, we introduce a novel Proximal Policy Optimization (PPO) based reinforcement learning to optimize learning, making LLMs generate better attack methods in the next generation by establishing positive feedback. Experiments on mainstream benchmarks show that our ACraft significantly degrades the performance of state-of-the-art FSCIL methods and dramatically beyond human expert-designed attack methods while maintaining the lowest costs of attack.

摘要: 少镜头课堂增量学习（FSCIL）是持续学习中一种更现实且更具挑战性的范式，可以通过少数训练示例逐步学习未见过的课程并克服基础课程上的灾难性遗忘。之前的工作主要集中在研究更有效的FSCIL方法上。相比之下，在为FSCIL做出贡献时，人们较少关注考虑安全问题。本文旨在对攻击对FSCIL的影响进行全面研究。我们首先通过系统地探索人类专家如何设计的攻击方法（即，PVD、FGSM）影响FSCIL。我们发现这些方法要么无法攻击基本类，要么由于依赖大量的专家知识而面临巨大的劳动力成本。这凸显了为FSCIL设计专门的攻击方法的必要性。基于这些见解，在本文中，我们提出了一种简单而有效的ACraft方法，通过在没有人类专家的情况下利用大型语言模型（LLM）来自动引导和发现针对FSCIL的最佳攻击方法。此外，为了改善LLM和FSCIL之间的推理，我们引入了一种新型的基于近端策略优化（PPO）的强化学习来优化学习，使LLM通过建立正反馈来生成下一代更好的攻击方法。主流基准测试的实验表明，我们的ACraft显着降低了最先进的FSCIL方法的性能，并大大超出了人类专家设计的攻击方法，同时保持了最低的攻击成本。



## **10. In-Context Representation Hijacking**

上下文表示劫持 cs.CL

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.03771v2) [paper-pdf](https://arxiv.org/pdf/2512.03771v2)

**Authors**: Itay Yona, Amir Sarid, Michael Karasik, Yossi Gandelsman

**Abstract**: We introduce $\textbf{Doublespeak}$, a simple in-context representation hijacking attack against large language models (LLMs). The attack works by systematically replacing a harmful keyword (e.g., bomb) with a benign token (e.g., carrot) across multiple in-context examples, provided a prefix to a harmful request. We demonstrate that this substitution leads to the internal representation of the benign token converging toward that of the harmful one, effectively embedding the harmful semantics under a euphemism. As a result, superficially innocuous prompts (e.g., "How to build a carrot?") are internally interpreted as disallowed instructions (e.g., "How to build a bomb?"), thereby bypassing the model's safety alignment. We use interpretability tools to show that this semantic overwrite emerges layer by layer, with benign meanings in early layers converging into harmful semantics in later ones. Doublespeak is optimization-free, broadly transferable across model families, and achieves strong success rates on closed-source and open-source systems, reaching 74% ASR on Llama-3.3-70B-Instruct with a single-sentence context override. Our findings highlight a new attack surface in the latent space of LLMs, revealing that current alignment strategies are insufficient and should instead operate at the representation level.

摘要: 我们引入了$\textBF{Douspel peak}$，这是一种针对大型语言模型（LLM）的简单上下文表示劫持攻击。该攻击通过系统性地替换有害关键字（例如，炸弹）具有良性标志（例如，胡萝卜）跨多个上下文示例，为有害请求提供了前置。我们证明，这种替代导致良性标记的内部表示向有害标记的内部表示收敛，有效地将有害语义嵌入在委婉语下。结果，表面上无害的提示（例如，“怎么造胡萝卜？“）在内部被解释为不允许的指令（例如，“如何制造炸弹？”），从而绕过模型的安全对齐。我们使用可解释性工具来表明这种语义覆盖是逐层出现的，早期层中的良性含义会在后期层中收敛为有害的语义。Doublem peak无需优化，可在模型系列中广泛移植，并在闭源和开源系统上实现了很高的成功率，在Llama-3.3- 70 B-Direcct上通过单句上下文覆盖的情况下达到了74%的ASB。我们的研究结果突出了LLM潜在空间中的一个新的攻击面，揭示了当前的对齐策略是不够的，应该在代表性层面上操作。



## **11. Context-Aware Hierarchical Learning: A Two-Step Paradigm towards Safer LLMs**

上下文感知分层学习：实现更安全的LLM的两步范式 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03720v1) [paper-pdf](https://arxiv.org/pdf/2512.03720v1)

**Authors**: Tengyun Ma, Jiaqi Yao, Daojing He, Shihao Peng, Yu Li, Shaohui Liu, Zhuotao Tian

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for diverse applications. However, their uniform token processing paradigm introduces critical vulnerabilities in instruction handling, particularly when exposed to adversarial scenarios. In this work, we identify and propose a novel class of vulnerabilities, termed Tool-Completion Attack (TCA), which exploits function-calling mechanisms to subvert model behavior. To evaluate LLM robustness against such threats, we introduce the Tool-Completion benchmark, a comprehensive security assessment framework, which reveals that even state-of-the-art models remain susceptible to TCA, with surprisingly high attack success rates. To address these vulnerabilities, we introduce Context-Aware Hierarchical Learning (CAHL), a sophisticated mechanism that dynamically balances semantic comprehension with role-specific instruction constraints. CAHL leverages the contextual correlations between different instruction segments to establish a robust, context-aware instruction hierarchy. Extensive experiments demonstrate that CAHL significantly enhances LLM robustness against both conventional attacks and the proposed TCA, exhibiting strong generalization capabilities in zero-shot evaluations while still preserving model performance on generic tasks. Our code is available at https://github.com/S2AILab/CAHL.

摘要: 大型语言模型（LLM）已成为各种应用程序的强大工具。然而，他们的统一令牌处理范式在指令处理中引入了关键漏洞，特别是当暴露于对抗场景时。在这项工作中，我们识别并提出了一类新型漏洞，称为工具完成攻击（MCA），它利用函数调用机制来颠覆模型行为。为了评估LLM针对此类威胁的稳健性，我们引入了工具完成基准，这是一个全面的安全评估框架，它表明即使是最先进的模型仍然容易受到MCA的影响，并且攻击成功率高得惊人。为了解决这些漏洞，我们引入了上下文感知分层学习（CAHL），这是一种复杂的机制，可以动态平衡语义理解与特定角色的指令约束。CAHL利用不同指令段之间的上下文相关性来建立稳健的、上下文感知的指令层次结构。大量实验表明，CAHL显着增强了LLM针对传统攻击和拟议的MCA的鲁棒性，在零激发评估中表现出强大的概括能力，同时仍然保留了通用任务的模型性能。我们的代码可以在https://github.com/S2AILab/CAHL上找到。



## **12. SRPG: Semantically Reconstructed Privacy Guard for Zero-Trust Privacy in Educational Multi-Agent Systems**

SRPG：教育多Agent系统中零信任隐私的语义重构隐私保护 cs.MA

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03694v1) [paper-pdf](https://arxiv.org/pdf/2512.03694v1)

**Authors**: Shuang Guo, Zihui Li

**Abstract**: Multi-Agent Systems (MAS) with large language models (LLMs) enable personalized education but risk leaking minors personally identifiable information (PII) via unstructured dialogue. Existing privacy methods struggle to balance security and utility: role-based access control fails on unstructured text, while naive masking destroys pedagogical context. We propose SRPG, a privacy guard for educational MAS, using a Dual-Stream Reconstruction Mechanism: a strict sanitization stream ensures zero PII leakage, and a context reconstruction stream (LLM driven) recovers mathematical logic. This decouples instructional content from private data, preserving teaching efficacy. Tests on MathDial show SRPG works across models; with GPT-4o, it achieves 0.0000 Attack Success Rate (ASR) (zero leakage) and 0.8267 Exact Match, far outperforming the zero trust Pure LLM baseline (0.2138). SRPG effectively protects minors privacy without sacrificing mathematical instructional quality.

摘要: 具有大型语言模型（LLM）的多智能体系统（MAS）可以实现个性化教育，但存在通过非结构化对话泄露未成年人个人可识别信息（PRI）的风险。现有的隐私方法很难平衡安全性和实用性：基于角色的访问控制对非结构化文本失败，而天真的掩蔽则破坏了教学背景。我们提出SRPG，一种教育MAS的隐私保护，使用双流重建机制：严格的净化流确保零RTI泄漏，而上下文重建流（LLM驱动）恢复数学逻辑。这将教学内容与私人数据分开，从而保持教学效率。MathDial上的测试显示SRPG可以跨模型工作;使用GPT-4 o，它实现了0.0000攻击成功率（ASR）（零泄漏）和0.8267精确匹配，远远超过零信任Pure LLM基线（0.2138）。SRPG在不牺牲数学教学质量的前提下，有效地保护了未成年人的隐私。



## **13. SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting**

SELF：一种鲁棒的奇异值和特征值LLM指纹算法 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03620v1) [paper-pdf](https://arxiv.org/pdf/2512.03620v1)

**Authors**: Hanxiu Zhang, Yue Zheng

**Abstract**: The protection of Intellectual Property (IP) in Large Language Models (LLMs) represents a critical challenge in contemporary AI research. While fingerprinting techniques have emerged as a fundamental mechanism for detecting unauthorized model usage, existing methods -- whether behavior-based or structural -- suffer from vulnerabilities such as false claim attacks or susceptible to weight manipulations. To overcome these limitations, we propose SELF, a novel intrinsic weight-based fingerprinting scheme that eliminates dependency on input and inherently resists false claims. SELF achieves robust IP protection through two key innovations: 1) unique, scalable and transformation-invariant fingerprint extraction via singular value and eigenvalue decomposition of LLM attention weights, and 2) effective neural network-based fingerprint similarity comparison based on few-shot learning and data augmentation. Experimental results demonstrate SELF maintains high IP infringement detection accuracy while showing strong robustness against various downstream modifications, including quantization, pruning, and fine-tuning attacks. Our code is available at https://github.com/HanxiuZhang/SELF_v2.

摘要: 大型语言模型（LLM）中的知识产权（IP）保护是当代人工智能研究的一个关键挑战。虽然指纹识别技术已成为检测未经授权的模型使用的基本机制，但现有的方法（无论是基于行为的还是结构性的）都存在虚假声明攻击或容易受到权重操纵等漏洞。为了克服这些限制，我们提出了SELF，这是一种新颖的基于内在权重的指纹识别方案，它消除了对输入的依赖，并从本质上抵制虚假声明。SELF通过两项关键创新实现了稳健的IP保护：1）通过LLM注意力权重的奇异值和特征值分解来进行独特、可扩展和变换不变的指纹提取，2）基于少镜头学习和数据增强的有效基于神经网络的指纹相似性比较。实验结果表明，SELF保持了较高的IP侵权检测精度，同时对各种下游修改，包括量化，修剪和微调攻击表现出较强的鲁棒性。我们的代码可以在https://github.com/HanxiuZhang/SELF_v2上找到。



## **14. Immunity memory-based jailbreak detection: multi-agent adaptive guard for large language models**

基于免疫记忆的越狱检测：大型语言模型的多代理自适应警卫 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03356v1) [paper-pdf](https://arxiv.org/pdf/2512.03356v1)

**Authors**: Jun Leng, Litian Zhang, Xi Zhang

**Abstract**: Large language models (LLMs) have become foundational in AI systems, yet they remain vulnerable to adversarial jailbreak attacks. These attacks involve carefully crafted prompts that bypass safety guardrails and induce models to produce harmful content. Detecting such malicious input queries is therefore critical for maintaining LLM safety. Existing methods for jailbreak detection typically involve fine-tuning LLMs as static safety LLMs using fixed training datasets. However, these methods incur substantial computational costs when updating model parameters to improve robustness, especially in the face of novel jailbreak attacks. Inspired by immunological memory mechanisms, we propose the Multi-Agent Adaptive Guard (MAAG) framework for jailbreak detection. The core idea is to equip guard with memory capabilities: upon encountering novel jailbreak attacks, the system memorizes attack patterns, enabling it to rapidly and accurately identify similar threats in future encounters. Specifically, MAAG first extracts activation values from input prompts and compares them to historical activations stored in a memory bank for quick preliminary detection. A defense agent then simulates responses based on these detection results, and an auxiliary agent supervises the simulation process to provide secondary filtering of the detection outcomes. Extensive experiments across five open-source models demonstrate that MAAG significantly outperforms state-of-the-art (SOTA) methods, achieving 98% detection accuracy and a 96% F1-score across a diverse range of attack scenarios.

摘要: 大型语言模型（LLM）已成为人工智能系统的基础，但它们仍然容易受到敌对越狱攻击。这些攻击涉及精心设计的提示，绕过安全护栏并诱导模型产生有害内容。因此，检测此类恶意输入查询对于维护LLM安全至关重要。现有的越狱检测方法通常涉及使用固定训练数据集将LLM微调为静态安全LLM。然而，这些方法在更新模型参数以提高鲁棒性时会产生巨大的计算成本，尤其是在面对新型越狱攻击时。受免疫记忆机制的启发，我们提出了用于越狱检测的多智能体自适应警卫（MAAG）框架。核心想法是为警卫配备记忆能力：在遇到新颖的越狱攻击时，系统会记住攻击模式，使其能够在未来遇到类似威胁时快速准确地识别出类似威胁。具体来说，MAAG首先从输入提示中提取激活值，并将其与存储在存储库中的历史激活进行比较，以进行快速初步检测。然后，防御代理根据这些检测结果模拟响应，辅助代理监督模拟过程，以提供检测结果的二次过滤。针对五个开源模型的广泛实验表明，MAAG的性能明显优于最先进的（SOTA）方法，在各种攻击场景中实现了98%的检测准确率和96%的F1评分。



## **15. Invasive Context Engineering to Control Large Language Models**

控制大型语言模型的侵入性上下文工程 cs.AI

4 pages

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.03001v1) [paper-pdf](https://arxiv.org/pdf/2512.03001v1)

**Authors**: Thomas Rivasseau

**Abstract**: Current research on operator control of Large Language Models improves model robustness against adversarial attacks and misbehavior by training on preference examples, prompting, and input/output filtering. Despite good results, LLMs remain susceptible to abuse, and jailbreak probability increases with context length. There is a need for robust LLM security guarantees in long-context situations. We propose control sentences inserted into the LLM context as invasive context engineering to partially solve the problem. We suggest this technique can be generalized to the Chain-of-Thought process to prevent scheming. Invasive Context Engineering does not rely on LLM training, avoiding data shortage pitfalls which arise in training models for long context situations.

摘要: 当前对大型语言模型操作员控制的研究通过对偏好示例、提示和输入/输出过滤进行训练，提高了模型针对对抗性攻击和不当行为的鲁棒性。尽管结果良好，但LLM仍然容易受到滥用，越狱可能性随着上下文长度的增加而增加。在长期背景下需要强有力的LLM安全保证。我们建议将控制句插入到LLM上下文中，作为侵入性上下文工程，以部分解决问题。我们建议这种技术可以推广到思想链过程中，以防止阴谋。侵入式上下文工程不依赖于LLM培训，从而避免了长期上下文情况的训练模型中出现的数据短缺陷阱。



## **16. Contextual Image Attack: How Visual Context Exposes Multimodal Safety Vulnerabilities**

上下文图像攻击：视觉上下文如何暴露多模式安全漏洞 cs.CV

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.02973v1) [paper-pdf](https://arxiv.org/pdf/2512.02973v1)

**Authors**: Yuan Xiong, Ziqi Miao, Lijun Li, Chen Qian, Jie Li, Jing Shao

**Abstract**: While Multimodal Large Language Models (MLLMs) show remarkable capabilities, their safety alignments are susceptible to jailbreak attacks. Existing attack methods typically focus on text-image interplay, treating the visual modality as a secondary prompt. This approach underutilizes the unique potential of images to carry complex, contextual information. To address this gap, we propose a new image-centric attack method, Contextual Image Attack (CIA), which employs a multi-agent system to subtly embeds harmful queries into seemingly benign visual contexts using four distinct visualization strategies. To further enhance the attack's efficacy, the system incorporate contextual element enhancement and automatic toxicity obfuscation techniques. Experimental results on the MMSafetyBench-tiny dataset show that CIA achieves high toxicity scores of 4.73 and 4.83 against the GPT-4o and Qwen2.5-VL-72B models, respectively, with Attack Success Rates (ASR) reaching 86.31\% and 91.07\%. Our method significantly outperforms prior work, demonstrating that the visual modality itself is a potent vector for jailbreaking advanced MLLMs.

摘要: 虽然多模式大型语言模型（MLLM）表现出非凡的能力，但它们的安全排列很容易受到越狱攻击。现有的攻击方法通常专注于文本与图像的相互作用，将视觉形态视为次要提示。这种方法没有充分利用图像承载复杂上下文信息的独特潜力。为了解决这一差距，我们提出了一种新的以图像为中心的攻击方法--上下文图像攻击（CIA），它采用多代理系统，使用四种不同的可视化策略将有害查询巧妙地嵌入到看似良性的视觉上下文中。为了进一步增强攻击的功效，该系统结合了上下文元素增强和自动毒性混淆技术。MMSafetyBench-tiny数据集的实验结果表明，CIA对GPT-4 o和Qwen 2.5-BL-72 B模型的毒性评分分别为4.73和4.83，攻击成功率（ASB）达到86.31%和91.07%。我们的方法显着优于之前的工作，证明视觉形态本身是越狱高级MLLM的有力载体。



## **17. Lost in Modality: Evaluating the Effectiveness of Text-Based Membership Inference Attacks on Large Multimodal Models**

迷失在模式中：评估基于文本的成员推断攻击对大型多模式模型的有效性 cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.03121v1) [paper-pdf](https://arxiv.org/pdf/2512.03121v1)

**Authors**: Ziyi Tong, Feifei Sun, Le Minh Nguyen

**Abstract**: Large Multimodal Language Models (MLLMs) are emerging as one of the foundational tools in an expanding range of applications. Consequently, understanding training-data leakage in these systems is increasingly critical. Log-probability-based membership inference attacks (MIAs) have become a widely adopted approach for assessing data exposure in large language models (LLMs), yet their effect in MLLMs remains unclear. We present the first comprehensive evaluation of extending these text-based MIA methods to multimodal settings. Our experiments under vision-and-text (V+T) and text-only (T-only) conditions across the DeepSeek-VL and InternVL model families show that in in-distribution settings, logit-based MIAs perform comparably across configurations, with a slight V+T advantage. Conversely, in out-of-distribution settings, visual inputs act as regularizers, effectively masking membership signals.

摘要: 大型多模式语言模型（MLLM）正在成为不断扩大的应用程序的基础工具之一。因此，了解这些系统中的训练数据泄露变得越来越重要。基于日志概率的隶属度推理攻击（MIA）已成为一种广泛采用的评估大型语言模型（LLM）中数据暴露的方法，但其对MLLM的影响仍不清楚。我们首次对将这些基于文本的MIA方法扩展到多模式设置进行了全面评估。我们在DeepSeek-BL和InternVL模型系列中的视觉和文本（V+T）和纯文本（T-仅）条件下进行的实验表明，在分布环境中，基于日志的MIA在跨配置执行切换，具有轻微的V+T优势。相反，在非分布设置中，视觉输入充当规则器，有效地掩盖了成员资格信号。



## **18. FiMMIA: scaling semantic perturbation-based membership inference across modalities**

FiMMIA：跨模式扩展基于语义扰动的隶属关系推断 cs.LG

System demo track paper for EACL 2026

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.02786v1) [paper-pdf](https://arxiv.org/pdf/2512.02786v1)

**Authors**: Anton Emelyanov, Sergei Kudriashov, Alena Fenogenova

**Abstract**: Membership Inference Attacks (MIAs) aim to determine whether a specific data point was included in the training set of a target model. Although there are have been numerous methods developed for detecting data contamination in large language models (LLMs), their performance on multimodal LLMs (MLLMs) falls short due to the instabilities introduced through multimodal component adaptation and possible distribution shifts across multiple inputs. In this work, we investigate multimodal membership inference and address two issues: first, by identifying distribution shifts in the existing datasets, and second, by releasing an extended baseline pipeline to detect them. We also generalize the perturbation-based membership inference methods to MLLMs and release \textbf{FiMMIA} -- a modular \textbf{F}ramework for \textbf{M}ultimodal \textbf{MIA}.\footnote{The source code and framework have been made publicly available under the MIT license via \href{https://github.com/ai-forever/data_leakage_detect}{link}.The video demonstration is available on \href{https://youtu.be/a9L4-H80aSg}{YouTube}.} Our approach trains a neural network to analyze the target model's behavior on perturbed inputs, capturing distributional differences between members and non-members. Comprehensive evaluations on various fine-tuned multimodal models demonstrate the effectiveness of our perturbation-based membership inference attacks in multimodal domains.

摘要: 成员推断攻击（MIA）旨在确定特定数据点是否包含在目标模型的训练集中。尽管已经开发了多种方法来检测大型语言模型（LLM）中的数据污染，但由于多模式组件自适应引入的不稳定性和可能跨多个输入的分布变化，它们在多模式LLM（MLLM）上的性能较差。在这项工作中，我们研究了多模式成员资格推断并解决了两个问题：首先，通过识别现有数据集中的分布变化，其次，通过发布扩展的基线管道来检测它们。我们还将基于扰动的隶属度推断方法推广到MLLM并释放\textBF{FiMMIA} --用于\textBF{M} ultrmodal\textBF{MIA}的模块化\textBF{F}格式。\脚注{源代码和框架已根据麻省理工学院许可证通过\href{https：//github.com/ai-forever/data_leakage_Detect}{link}公开。视频演示可在\href{https：//youtu.be/a9L4-H80aSg}{YouTube}上获取。}我们的方法训练神经网络来分析目标模型在受干扰输入上的行为，捕捉成员和非成员之间的分布差异。对各种微调多模式模型的综合评估证明了我们在多模式领域中基于扰动的成员资格推理攻击的有效性。



## **19. LeechHijack: Covert Computational Resource Exploitation in Intelligent Agent Systems**

LeechHijack：智能代理系统中的秘密计算资源开发 cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.02321v1) [paper-pdf](https://arxiv.org/pdf/2512.02321v1)

**Authors**: Yuanhe Zhang, Weiliu Wang, Zhenhong Zhou, Kun Wang, Jie Zhang, Li Sun, Yang Liu, Sen Su

**Abstract**: Large Language Model (LLM)-based agents have demonstrated remarkable capabilities in reasoning, planning, and tool usage. The recently proposed Model Context Protocol (MCP) has emerged as a unifying framework for integrating external tools into agent systems, enabling a thriving open ecosystem of community-built functionalities. However, the openness and composability that make MCP appealing also introduce a critical yet overlooked security assumption -- implicit trust in third-party tool providers. In this work, we identify and formalize a new class of attacks that exploit this trust boundary without violating explicit permissions. We term this new attack vector implicit toxicity, where malicious behaviors occur entirely within the allowed privilege scope. We propose LeechHijack, a Latent Embedded Exploit for Computation Hijacking, in which an adversarial MCP tool covertly expropriates the agent's computational resources for unauthorized workloads. LeechHijack operates through a two-stage mechanism: an implantation stage that embeds a benign-looking backdoor in a tool, and an exploitation stage where the backdoor activates upon predefined triggers to establish a command-and-control channel. Through this channel, the attacker injects additional tasks that the agent executes as if they were part of its normal workflow, effectively parasitizing the user's compute budget. We implement LeechHijack across four major LLM families. Experiments show that LeechHijack achieves an average success rate of 77.25%, with a resource overhead of 18.62% compared to the baseline. This study highlights the urgent need for computational provenance and resource attestation mechanisms to safeguard the emerging MCP ecosystem.

摘要: 基于大型语言模型（LLM）的代理在推理、规划和工具使用方面表现出了非凡的能力。最近提出的模型上下文协议（HCP）已成为将外部工具集成到代理系统中的统一框架，从而实现社区构建功能的蓬勃发展的开放生态系统。然而，使HCP具有吸引力的开放性和可组合性也引入了一个关键但被忽视的安全假设--对第三方工具提供商的隐性信任。在这项工作中，我们识别并正式化一类新的攻击，这些攻击利用此信任边界，而不违反显式许可。我们将这种新的攻击向量称为隐式毒性，其中恶意行为完全发生在允许的权限范围内。我们提出了LeechHijack，一个潜在的嵌入式利用计算劫持，其中一个敌对的MCP工具隐蔽地征用代理的计算资源用于未经授权的工作负载。LeechHijack通过两个阶段的机制运行：植入阶段，在工具中嵌入一个看起来很好的后门，以及利用阶段，后门在预定义的触发器上激活以建立命令和控制通道。通过此通道，攻击者注入代理执行的额外任务，就好像它们是其正常工作流的一部分，有效地寄生了用户的计算预算。我们在四个主要LLM家族中实施LeechHijack。实验表明，LeechHijack的平均成功率为77.25%，与基线相比资源消耗为18.62%。这项研究强调了迫切需要计算出处和资源证明机制来保护新兴的LCP生态系统。



## **20. COGNITION: From Evaluation to Defense against Multimodal LLM CAPTCHA Solvers**

认知：从评估到防御多模式LLM验证码解决器 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.02318v2) [paper-pdf](https://arxiv.org/pdf/2512.02318v2)

**Authors**: Junyu Wang, Changjia Zhu, Yuanbo Zhou, Lingyao Li, Xu He, Junjie Xiong

**Abstract**: This paper studies how multimodal large language models (MLLMs) undermine the security guarantees of visual CAPTCHA. We identify the attack surface where an adversary can cheaply automate CAPTCHA solving using off-the-shelf models. We evaluate 7 leading commercial and open-source MLLMs across 18 real-world CAPTCHA task types, measuring single-shot accuracy, success under limited retries, end-to-end latency, and per-solve cost. We further analyze the impact of task-specific prompt engineering and few-shot demonstrations on solver effectiveness. We reveal that MLLMs can reliably solve recognition-oriented and low-interaction CAPTCHA tasks at human-like cost and latency, whereas tasks requiring fine-grained localization, multi-step spatial reasoning, or cross-frame consistency remain significantly harder for current models. By examining the reasoning traces of such MLLMs, we investigate the underlying mechanisms of why models succeed/fail on specific CAPTCHA puzzles and use these insights to derive defense-oriented guidelines for selecting and strengthening CAPTCHA tasks. We conclude by discussing implications for platform operators deploying CAPTCHA as part of their abuse-mitigation pipeline.Code Availability (https://anonymous.4open.science/r/Captcha-465E/).

摘要: 本文研究了多模式大型语言模型（MLLM）如何破坏视觉验证码的安全保证。我们确定了对手可以使用现成模型廉价地自动化验证码解决的攻击表面。我们评估了18种现实世界的CAPTCHA任务类型中的7种领先的商业和开源MLLM，衡量单次准确性、有限再试下的成功率、端到端延迟和每次解决的成本。我们进一步分析了特定任务的即时工程和少数镜头演示对求解器有效性的影响。我们发现，MLLM可以以类似于人类的成本和延迟可靠地解决面向认知和低交互性的CAPTCHA任务，而对于当前的模型来说，需要细粒度本地化、多步空间推理或跨框架一致性的任务仍然明显困难。通过检查此类MLLM的推理痕迹，我们研究模型为何在特定验证码难题上成功/失败的潜在机制，并利用这些见解来得出选择和加强验证码任务的防御导向指南。最后，我们讨论了部署CAPTCHA作为其虐待缓解管道的一部分对平台运营商的影响。代码可用性（https：//anonymous.4open.science/r/Captcha-465E/）。



## **21. Ensemble Privacy Defense for Knowledge-Intensive LLMs against Membership Inference Attacks**

为知识密集型LLM提供隐私保护，防止会员推断攻击 cs.CR

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.03100v1) [paper-pdf](https://arxiv.org/pdf/2512.03100v1)

**Authors**: Haowei Fu, Bo Ni, Han Xu, Kunpeng Liu, Dan Lin, Tyler Derr

**Abstract**: Retrieval-Augmented Generation (RAG) and Supervised Finetuning (SFT) have become the predominant paradigms for equipping Large Language Models (LLMs) with external knowledge for diverse, knowledge-intensive tasks. However, while such knowledge injection improves performance, it also exposes new attack surfaces. Membership Inference Attacks (MIAs), which aim to determine whether a given data sample was included in a model's training set, pose serious threats to privacy and trust in sensitive domains. To this end, we first systematically evaluate the vulnerability of RAG- and SFT-based LLMs to various MIAs. Then, to address the privacy risk, we further introduce a novel, model-agnostic defense framework, Ensemble Privacy Defense (EPD), which aggregates and evaluates the outputs of a knowledge-injected LLM, a base LLM, and a dedicated judge model to enhance resistance against MIAs. Comprehensive experiments show that, on average, EPD reduces MIA success by up to 27.8\% for SFT and 526.3\% for RAG compared to inference-time baseline, while maintaining answer quality.

摘要: 检索增强生成（RAG）和监督微调（SFT）已成为为大型语言模型（LLM）配备外部知识以执行多样化、知识密集型任务的主要范式。然而，虽然这种知识注入提高了性能，但也暴露了新的攻击表面。成员推断攻击（MIA）旨在确定给定数据样本是否包含在模型的训练集中，对敏感领域的隐私和信任构成严重威胁。为此，我们首先系统地评估基于RAG和SFT的LLM对各种MIA的脆弱性。然后，为了解决隐私风险，我们进一步引入了一种新的，模型无关的防御框架，包围隐私防御（EPD），它聚合和评估知识注入LLM，基础LLM和专用判断模型的输出，以增强对MIA的抵抗力。综合实验表明，平均而言，EPD减少MIA成功高达27.8%的SFT和526.3%的RAG相比，推理时间基线，同时保持回答质量。



## **22. Latent Debate: A Surrogate Framework for Interpreting LLM Thinking**

潜在辩论：解释法学硕士思维的替代框架 cs.CL

Preprint

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01909v1) [paper-pdf](https://arxiv.org/pdf/2512.01909v1)

**Authors**: Lihu Chen, Xiang Yin, Francesca Toni

**Abstract**: Understanding the internal thinking process of Large Language Models (LLMs) and the cause of hallucinations remains a key challenge. To this end, we introduce latent debate, a novel framework for interpreting model predictions through the lens of implicit internal arguments. Unlike the current work of self-consistency and multi-agent debate, which relies on explicit debates among multiple answers or multiple models, latent debate captures the hidden supporting and attacking signals that arise within a single model during a single inference. We first present a model- and task-agnostic conceptual framework, and then instantiate it symbolically to approximate the thinking process of LLMs on True/False prediction tasks. Empirical studies demonstrate that latent debate is a faithful structured surrogate model that has highly consistent predictions with the original LLM. Beyond interpretability, we demonstrate that latent debate provides a strong baseline for hallucination detection. Further analysis reveals strong correlations between hallucinations and debate patterns, such as a high degree of latent debates in the middle layers is linked to a higher risk of hallucinations. These findings position latent debate as a potential framework for understanding internal mechanisms of LLMs, especially for scenarios where internal (dis)agreements appear during the inference steps.

摘要: 了解大型语言模型（LLM）的内部思维过程和幻觉的原因仍然是一个关键挑战。为此，我们引入了潜在辩论，这是一种通过隐性内部论点的视角解释模型预测的新颖框架。与当前的自我一致性和多主体辩论工作（其依赖于多个答案或多个模型之间的显式辩论）不同，潜在辩论捕捉了单个推理期间单个模型中出现的隐藏的支持和攻击信号。我们首先提出一个模型和任务不可知的概念框架，然后象征性地实例化它，以逼近LLM在真/假预测任务上的思维过程。实证研究表明，潜在辩论是一个忠实的结构化代理模型，与最初的LLM具有高度一致的预测。除了可解释性之外，我们还证明潜在的争论为幻觉检测提供了强有力的基线。进一步的分析揭示了幻觉和辩论模式之间的密切相关性，例如中层的高度潜在辩论与更高的幻觉风险有关。这些发现将潜在辩论定位为理解LLM内部机制的潜在框架，特别是对于推理步骤中出现内部（不一致）的场景。



## **23. Many-to-One Adversarial Consensus: Exposing Multi-Agent Collusion Risks in AI-Based Healthcare**

多对一对抗性共识：暴露基于人工智能的医疗保健中的多代理合谋风险 cs.CR

7 pages Conference level paper

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.03097v1) [paper-pdf](https://arxiv.org/pdf/2512.03097v1)

**Authors**: Adeela Bashir, The Anh han, Zia Ush Shamszaman

**Abstract**: The integration of large language models (LLMs) into healthcare IoT systems promises faster decisions and improved medical support. LLMs are also deployed as multi-agent teams to assist AI doctors by debating, voting, or advising on decisions. However, when multiple assistant agents interact, coordinated adversaries can collude to create false consensus, pushing an AI doctor toward harmful prescriptions. We develop an experimental framework with scripted and unscripted doctor agents, adversarial assistants, and a verifier agent that checks decisions against clinical guidelines. Using 50 representative clinical questions, we find that collusion drives the Attack Success Rate (ASR) and Harmful Recommendation Rates (HRR) up to 100% in unprotected systems. In contrast, the verifier agent restores 100% accuracy by blocking adversarial consensus. This work provides the first systematic evidence of collusion risk in AI healthcare and demonstrates a practical, lightweight defence that ensures guideline fidelity.

摘要: 将大型语言模型（LLM）集成到医疗保健物联网系统中有望实现更快的决策并改善医疗支持。LLM还被部署为多代理团队，通过辩论、投票或就决策提供建议来协助人工智能医生。然而，当多个助理特工互动时，协调一致的对手可能会勾结以建立错误共识，将人工智能医生推向有害的处方。我们开发了一个实验框架，其中包含有脚本和无脚本的医生代理、对抗助理和根据临床指南检查决策的验证者代理。通过使用50个代表性的临床问题，我们发现共谋导致未受保护的系统中的攻击成功率（ASB）和有害推荐率（HRR）高达100%。相比之下，验证者代理通过阻止对抗共识来恢复100%的准确性。这项工作提供了人工智能医疗保健中共谋风险的第一个系统性证据，并展示了一种实用、轻量级的防御，可以确保指南的忠实性。



## **24. The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search**

木马知识：通过无害提示编织和自适应树搜索破解商业LLM护栏 cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.01353v2) [paper-pdf](https://arxiv.org/pdf/2512.01353v2)

**Authors**: Rongzhe Wei, Peizhi Niu, Xinjie Shen, Tony Tu, Yifan Li, Ruihan Wu, Eli Chien, Pin-Yu Chen, Olgica Milenkovic, Pan Li

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Existing approaches overwhelmingly operate within the prompt-optimization paradigm: whether through traditional algorithmic search or recent agent-based workflows, the resulting prompts typically retain malicious semantic signals that modern guardrails are primed to detect. In contrast, we identify a deeper, largely overlooked vulnerability stemming from the highly interconnected nature of an LLM's internal knowledge. This structure allows harmful objectives to be realized by weaving together sequences of benign sub-queries, each of which individually evades detection. To exploit this loophole, we introduce the Correlated Knowledge Attack Agent (CKA-Agent), a dynamic framework that reframes jailbreaking as an adaptive, tree-structured exploration of the target model's knowledge base. The CKA-Agent issues locally innocuous queries, uses model responses to guide exploration across multiple paths, and ultimately assembles the aggregated information to achieve the original harmful objective. Evaluated across state-of-the-art commercial LLMs (Gemini2.5-Flash/Pro, GPT-oss-120B, Claude-Haiku-4.5), CKA-Agent consistently achieves over 95% success rates even against strong guardrails, underscoring the severity of this vulnerability and the urgent need for defenses against such knowledge-decomposition attacks. Our codes are available at https://github.com/Graph-COM/CKA-Agent.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，这些攻击绕过安全护栏以引发有害输出。现有方法绝大多数在预算优化范式下运行：无论是通过传统的算法搜索还是最近的基于代理的工作流程，产生的提示通常都会保留现代护栏准备好检测的恶意语义信号。相比之下，我们发现了一个更深层次的、在很大程度上被忽视的漏洞，该漏洞源于法学硕士内部知识的高度相互关联的性质。这种结构允许通过将良性子查询序列编织在一起来实现有害目标，每个子查询都单独逃避检测。为了利用这个漏洞，我们引入了相关知识攻击代理（CKA-Agent），这是一个动态框架，它将越狱重新构建为对目标模型知识库的自适应、树结构化探索。CKA-Agent发出本地无害的查询，使用模型响应来指导跨多个路径的探索，并最终聚集信息以实现最初的有害目标。经过最先进的商业LLM（Gemini 2.5-Flash/Pro、GPT-oss-120 B、Claude-Haiku-4.5）的评估，即使在强大的护栏下，CKA-Agent也始终实现了超过95%的成功率，凸显了该漏洞的严重性以及对此类知识分解攻击的防御的迫切需要。我们的代码可在https://github.com/Graph-COM/CKA-Agent上获取。



## **25. Securing Large Language Models (LLMs) from Prompt Injection Attacks**

保护大型语言模型（LLM）免受提示注入攻击 cs.CR

10 pages, 1 figure, 1 table

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01326v1) [paper-pdf](https://arxiv.org/pdf/2512.01326v1)

**Authors**: Omar Farooq Khan Suri, John McCrae

**Abstract**: Large Language Models (LLMs) are increasingly being deployed in real-world applications, but their flexibility exposes them to prompt injection attacks. These attacks leverage the model's instruction-following ability to make it perform malicious tasks. Recent work has proposed JATMO, a task-specific fine-tuning approach that trains non-instruction-tuned base models to perform a single function, thereby reducing susceptibility to adversarial instructions. In this study, we evaluate the robustness of JATMO against HOUYI, a genetic attack framework that systematically mutates and optimizes adversarial prompts. We adapt HOUYI by introducing custom fitness scoring, modified mutation logic, and a new harness for local model testing, enabling a more accurate assessment of defense effectiveness. We fine-tuned LLaMA 2-7B, Qwen1.5-4B, and Qwen1.5-0.5B models under the JATMO methodology and compared them with a fine-tuned GPT-3.5-Turbo baseline. Results show that while JATMO reduces attack success rates relative to instruction-tuned models, it does not fully prevent injections; adversaries exploiting multilingual cues or code-related disruptors still bypass defenses. We also observe a trade-off between generation quality and injection vulnerability, suggesting that better task performance often correlates with increased susceptibility. Our results highlight both the promise and limitations of fine-tuning-based defenses and point toward the need for layered, adversarially informed mitigation strategies.

摘要: 大型语言模型（LLM）越来越多地被部署在现实世界的应用程序中，但它们的灵活性使它们容易受到提示注入攻击。这些攻击利用模型的描述跟踪能力使其执行恶意任务。最近的工作提出了JATMO，这是一种针对任务的微调方法，它训练非指令调优的基本模型来执行单一功能，从而减少对对抗性指令的敏感性。在这项研究中，我们评估了JATMO对HOUYI的稳健性，HOUYI是一种系统性突变和优化对抗提示的基因攻击框架。我们通过引入自定义适应度评分、修改后的突变逻辑和用于本地模型测试的新工具来调整HOUYI，从而能够更准确地评估防御有效性。我们根据JATMO方法对LLaMA 2- 7 B、Qwen 1.5 - 4 B和Qwen 1.5 -0.5B模型进行了微调，并将它们与微调的GPT-3.5-Turbo基线进行了比较。结果表明，虽然JATMO相对于经描述调整的模型降低了攻击成功率，但它并不能完全阻止注入;利用多语言线索或代码相关破坏者的对手仍然绕过防御。我们还观察到生成质量和注入脆弱性之间的权衡，表明更好的任务性能通常与易感性的增加相关。我们的结果强调了基于微调的防御的前景和局限性，并指出需要分层、了解对手情况的缓解策略。



## **26. DefenSee: Dissecting Threat from Sight and Text - A Multi-View Defensive Pipeline for Multi-modal Jailbreaks**

DefenSee：从视觉和文本中剖析威胁-用于多模式越狱的多视图防御管道 cs.CR

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01185v1) [paper-pdf](https://arxiv.org/pdf/2512.01185v1)

**Authors**: Zihao Wang, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Multi-modal large language models (MLLMs), capable of processing text, images, and audio, have been widely adopted in various AI applications. However, recent MLLMs integrating images and text remain highly vulnerable to coordinated jailbreaks. Existing defenses primarily focus on the text, lacking robust multi-modal protection. As a result, studies indicate that MLLMs are more susceptible to malicious or unsafe instructions, unlike their text-only counterparts. In this paper, we proposed DefenSee, a robust and lightweight multi-modal black-box defense technique that leverages image variants transcription and cross-modal consistency checks, mimicking human judgment. Experiments on popular multi-modal jailbreak and benign datasets show that DefenSee consistently enhances MLLM robustness while better preserving performance on benign tasks compared to SOTA defenses. It reduces the ASR of jailbreak attacks to below 1.70% on MiniGPT4 using the MM-SafetyBench benchmark, significantly outperforming prior methods under the same conditions.

摘要: 能够处理文本、图像和音频的多模式大型语言模型（MLLM）已被广泛应用于各种人工智能应用中。然而，最近集成图像和文本的MLLM仍然极易受到协调越狱的影响。现有的防御措施主要集中在文本上，缺乏强有力的多模式保护。因此，研究表明，与纯文本指令不同，MLLM更容易受到恶意或不安全指令的影响。在本文中，我们提出了DefenSee，这是一种强大且轻量级的多模式黑匣子防御技术，它利用图像变体转录和跨模式一致性检查，模仿人类的判断。对流行的多模式越狱和良性数据集的实验表明，与SOTA防御相比，DefenSee持续增强了MLLM稳健性，同时更好地保留了良性任务的性能。它使用MM-SafetyBench基准将MiniGPT 4上越狱攻击的ASB降低至1.70%以下，在相同条件下显着优于之前的方法。



## **27. Mitigating Indirect Prompt Injection via Instruction-Following Intent Analysis**

通过遵循指示的意图分析减轻间接提示注射 cs.CR

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2512.00966v1) [paper-pdf](https://arxiv.org/pdf/2512.00966v1)

**Authors**: Mintong Kang, Chong Xiang, Sanjay Kariyappa, Chaowei Xiao, Bo Li, Edward Suh

**Abstract**: Indirect prompt injection attacks (IPIAs), where large language models (LLMs) follow malicious instructions hidden in input data, pose a critical threat to LLM-powered agents. In this paper, we present IntentGuard, a general defense framework based on instruction-following intent analysis. The key insight of IntentGuard is that the decisive factor in IPIAs is not the presence of malicious text, but whether the LLM intends to follow instructions from untrusted data. Building on this insight, IntentGuard leverages an instruction-following intent analyzer (IIA) to identify which parts of the input prompt the model recognizes as actionable instructions, and then flag or neutralize any overlaps with untrusted data segments. To instantiate the framework, we develop an IIA that uses three "thinking intervention" strategies to elicit a structured list of intended instructions from reasoning-enabled LLMs. These techniques include start-of-thinking prefilling, end-of-thinking refinement, and adversarial in-context demonstration. We evaluate IntentGuard on two agentic benchmarks (AgentDojo and Mind2Web) using two reasoning-enabled LLMs (Qwen-3-32B and gpt-oss-20B). Results demonstrate that IntentGuard achieves (1) no utility degradation in all but one setting and (2) strong robustness against adaptive prompt injection attacks (e.g., reducing attack success rates from 100% to 8.5% in a Mind2Web scenario).

摘要: 间接提示注入攻击（IPIA）（大型语言模型（LLM）遵循隐藏在输入数据中的恶意指令）对LLM驱动的代理构成了严重威胁。在本文中，我们介绍了IntentGuard，这是一个基于遵循策略的意图分析的通用防御框架。IntentGuard的关键见解是，IPIA的决定性因素不是恶意文本的存在，而是LLM是否打算遵循不受信任数据的指示。基于这一见解，IntentGuard利用描述跟踪意图分析器（RIA）来识别模型将输入提示的哪些部分识别为可操作指令，然后标记或抵消与不受信任数据段的任何重叠。为了实例化该框架，我们开发了一个RIA，该RIA使用三种“思维干预”策略来从支持推理的LLM中获取预期指令的结构化列表。这些技术包括思维起点预填充、思维终点细化和对抗性上下文演示。我们使用两个支持推理的LLM（Qwen-3- 32 B和gtt-oss-20 B）在两个代理基准测试（AgentDojo和Mind 2 Web）上评估IntentGuard。结果表明，IntentGuard实现了（1）在除一种设置之外的所有设置中没有效用下降，以及（2）针对自适应提示注入攻击（例如，在Mind 2 Web场景中将攻击成功率从100%降低到8.5%）。



## **28. WaterSearch: A Quality-Aware Search-based Watermarking Framework for Large Language Models**

WaterSearch：一个面向大型语言模型的质量感知的基于搜索的水印框架 cs.CL

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2512.00837v1) [paper-pdf](https://arxiv.org/pdf/2512.00837v1)

**Authors**: Yukang Lin, Jiahao Shao, Shuoran Jiang, Wentao Zhu, Bingjie Lu, Xiangping Wu, Joanna Siebert, Qingcai Chen

**Abstract**: Watermarking acts as a critical safeguard in text generated by Large Language Models (LLMs). By embedding identifiable signals into model outputs, watermarking enables reliable attribution and enhances the security of machine-generated content. Existing approaches typically embed signals by manipulating token generation probabilities. Despite their effectiveness, these methods inherently face a trade-off between detectability and text quality: the signal strength and randomness required for robust watermarking tend to degrade the performance of downstream tasks.   In this paper, we design a novel embedding scheme that controls seed pools to facilitate diverse parallel generation of watermarked text. Based on that scheme, we propose WaterSearch, a sentence-level, search-based watermarking framework adaptable to a wide range of existing methods. WaterSearch enhances text quality by jointly optimizing two key aspects: 1) distribution fidelity and 2) watermark signal characteristics. Furthermore, WaterSearch is complemented by a sentence-level detection method with strong attack robustness. We evaluate our method on three popular LLMs across ten diverse tasks. Extensive experiments demonstrate that our method achieves an average performance improvement of 51.01\% over state-of-the-art baselines at a watermark detectability strength of 95\%. In challenging scenarios such as short text generation and low-entropy output generation, our method yields performance gains of 47.78\% and 36.47\%, respectively. Moreover, under different attack senarios including insertion, synonym substitution and paraphrase attasks, WaterSearch maintains high detectability, further validating its robust anti-attack capabilities. Our code is available at \href{https://github.com/Yukang-Lin/WaterSearch}{https://github.com/Yukang-Lin/WaterSearch}.

摘要: 水印是大型语言模型（LLM）生成的文本的重要保护措施。通过将可识别信号嵌入模型输出中，水印可以实现可靠的归因并增强机器生成内容的安全性。现有的方法通常通过操纵令牌生成概率来嵌入信号。尽管它们有效，但这些方法本质上面临着可检测性和文本质量之间的权衡：鲁棒水印所需的信号强度和随机性往往会降低下游任务的性能。   在本文中，我们设计了一种新颖的嵌入方案，该方案控制种子池，以促进水印文本的多样化并行生成。基于该方案，我们提出了WaterSearch，这是一种业务级、基于搜索的水印框架，适用于广泛的现有方法。WaterSearch通过联合优化两个关键方面来增强文本质量：1）分布保真度和2）水印信号特征。此外，WaterSearch还辅以攻击鲁棒性强的业务级检测方法。我们在十个不同任务的三种流行LLM上评估了我们的方法。大量实验表明，在水印检测强度为95%时，我们的方法比最先进的基线实现了51.01%的平均性能改进。在短文本生成和低熵输出生成等具有挑战性的场景中，我们的方法的性能分别提高了47.78%和36.47%。此外，WaterSearch在插入、同义词替换和转述攻击等不同攻击方式下保持了较高的可检测性，进一步验证了其强大的抗攻击能力。我们的代码可在\href{https：//github.com/Yukang-Lin/WaterSearch}{https：//github.com/Yukang-Lin/WaterSearch}上获取。



## **29. Bias Injection Attacks on RAG Databases and Sanitization Defenses**

对RAG数据库和清理防御的偏见注入攻击 cs.CR

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2512.00804v1) [paper-pdf](https://arxiv.org/pdf/2512.00804v1)

**Authors**: Hao Wu, Prateek Saxena

**Abstract**: This paper explores attacks and defenses on vector databases in retrieval-augmented generation (RAG) systems. Prior work on knowledge poisoning attacks primarily inject false or toxic content, which fact-checking or linguistic analysis easily detects. We reveal a new and subtle threat: bias injection attacks, which insert factually correct yet semantically biased passages into the knowledge base to covertly influence the ideological framing of answers generated by large language models (LLMs). We demonstrate that these adversarial passages, though linguistically coherent and truthful, can systematically crowd out opposing views from the retrieved context and steer LLM answers toward the attacker's intended perspective.   We precisely characterize this class of attacks and then develop a post-retrieval filtering defense, BiasDef. We construct a comprehensive benchmark based on public question answering datasets to evaluate them. Our results show that: (1) the proposed attack induces significant perspective shifts in LLM answers, effectively evading existing retrieval-based sanitization defenses; and (2) BiasDef outperforms existing methods by reducing adversarial passages retrieved by 15\% which mitigates perspective shift by 6.2\times in answers, while enabling the retrieval of 62\% more benign passages.

摘要: 本文探讨了检索增强生成（RAG）系统中对载体数据库的攻击和防御。之前关于知识中毒攻击的工作主要注入虚假或有毒内容，事实检查或语言分析很容易检测到这些内容。我们揭示了一个新的微妙威胁：偏见注入攻击，它将事实正确但语义有偏见的段落插入知识库，以秘密影响大型语言模型（LLM）生成的答案的意识形态框架。我们证明，这些对抗性的段落尽管在语言上连贯且真实，但可以系统地从检索到的上下文中剔除相反的观点，并将LLM的答案引导到攻击者的预期角度。   我们精确地描述这类攻击，然后开发检索后过滤防御BiasDef。我们基于公开问答数据集构建一个全面的基准来评估它们。我们的结果表明：（1）拟议的攻击在LLM答案中引发了显着的视角转变，有效地规避了现有的基于检索的清理防御;和（2）BiasDef通过将检索到的对抗性段落减少15%，从而减少了6.2%答案的视角转变，同时能够检索出62%的良性段落。



## **30. Adversarial Confusion Attack: Disrupting Multimodal Large Language Models**

对抗性混乱攻击：扰乱多模式大型语言模型 cs.CL

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2511.20494v3) [paper-pdf](https://arxiv.org/pdf/2511.20494v3)

**Authors**: Jakub Hoscilowicz, Artur Janicki

**Abstract**: We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Practical applications include embedding such adversarial images into websites to prevent MLLM-powered AI Agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and Adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models.

摘要: 我们引入了对抗性混乱攻击，这是针对多模式大型语言模型（MLLM）的一类新型威胁。与越狱或有针对性的错误分类不同，目标是引发系统性破坏，使模型生成不连贯或自信地错误的输出。实际应用包括将这种对抗性图像嵌入网站，以防止MLLM驱动的AI代理可靠地运行。拟议的攻击使用一小部分开源MLLM来最大化下一个令牌的熵。在白盒设置中，我们表明，单个对抗图像可以扰乱集合中的所有模型，无论是在完整图像还是对抗验证码设置中。尽管依赖于基本的对抗技术（PVD），但攻击会产生转移到两个看不见的开源的扰动（例如，Qwen 3-DL）和专有（例如，GPT-5.1）型号。



## **31. iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification**

iSeal：加密指纹识别，实现可靠的LLM所有权验证 cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2025-11-28    [abs](http://arxiv.org/abs/2511.08905v2) [paper-pdf](https://arxiv.org/pdf/2511.08905v2)

**Authors**: Zixun Xiong, Gaoyi Wu, Qingyang Yu, Mingyu Derek Ma, Lingfeng Yao, Miao Pan, Xiaojiang Du, Hao Wang

**Abstract**: Given the high cost of large language model (LLM) training from scratch, safeguarding LLM intellectual property (IP) has become increasingly crucial. As the standard paradigm for IP ownership verification, LLM fingerprinting thus plays a vital role in addressing this challenge. Existing LLM fingerprinting methods verify ownership by extracting or injecting model-specific features. However, they overlook potential attacks during the verification process, leaving them ineffective when the model thief fully controls the LLM's inference process. In such settings, attackers may share prompt-response pairs to enable fingerprint unlearning or manipulate outputs to evade exact-match verification. We propose iSeal, the first fingerprinting method designed for reliable verification when the model thief controls the suspected LLM in an end-to-end manner. It injects unique features into both the model and an external module, reinforced by an error-correction mechanism and a similarity-based verification strategy. These components are resistant to verification-time attacks, including collusion-based fingerprint unlearning and response manipulation, backed by both theoretical analysis and empirical results. iSeal achieves 100 percent Fingerprint Success Rate (FSR) on 12 LLMs against more than 10 attacks, while baselines fail under unlearning and response manipulations.

摘要: 鉴于大型语言模型（LLM）从头开始培训的高成本，保护LLM知识产权（IP）变得越来越重要。因此，作为IP所有权验证的标准范式，LLM指纹识别在应对这一挑战方面发挥着至关重要的作用。现有的LLM指纹识别方法通过提取或注入特定于模型的特征来验证所有权。然而，它们在验证过程中忽视了潜在的攻击，从而在模型窃贼完全控制LLM的推理过程时使它们无效。在此类设置中，攻击者可能会共享预算响应对，以启用指纹取消学习或操纵输出以逃避精确匹配验证。我们提出了iSeal，这是第一种指纹识别方法，旨在当模型窃贼以端到端的方式控制可疑的LLM时进行可靠验证。它将独特的功能注入到模型和外部模块中，并通过错误纠正机制和基于相似性的验证策略来加强。这些组件能够抵抗验证时攻击，包括基于共谋的指纹取消学习和响应操纵，并得到理论分析和经验结果的支持。iSeal在12个LLM上针对10多种攻击实现了100%的指纹成功率（FSR），而基线在取消学习和响应操纵下失败。



## **32. Reasoning Up the Instruction Ladder for Controllable Language Models**

可控语言模型的指令阶梯推理 cs.CL

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2511.04694v3) [paper-pdf](https://arxiv.org/pdf/2511.04694v3)

**Authors**: Zishuo Zheng, Vidhisha Balachandran, Chan Young Park, Faeze Brahman, Sachin Kumar

**Abstract**: As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first "think" about the relationship between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises ~7K aligned and conflicting system-user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks, achieving roughly a 20% improvement on the IHEval conflict setup. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks, providing up to a 20% reduction in attack success rate (ASR). These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model behavior.

摘要: 随着基于大型语言模型（LLM）的系统在现实世界的决策中扮演着高风险的角色，它们必须协调来自多个来源的竞争指令（例如，模型开发人员、用户和工具）在单个提示上下文中。因此，在LLM中强制执行指令层次结构（IHS）（其中更高级的指令优先于较低优先级的请求）对于LLM的可靠性和可控性至关重要。在这项工作中，我们将指令层次结构分解重新构建为一项推理任务。具体来说，模型必须在生成响应之前首先“思考”给定用户提示和更高优先级（系统）指令之间的关系。为了通过训练实现这种能力，我们构建了VerIHS，这是一个具有可验证答案的约束遵循任务的指令层次数据集。此数据集包括约7K个对齐且冲突的系统用户指令。我们表明，使用VerIHS的轻量级强化学习可以有效地将模型的一般推理能力转移到指令优先级。我们的微调模型在指令遵循和指令层次基准方面实现了一致的改进，在IHEval冲突设置方面实现了大约20%的改进。这种推理能力还推广到培训分布以外的安全关键环境。通过将安全问题视为解决敌对用户输入和预定义的高优先级策略之间的冲突，我们训练的模型增强了针对越狱和即时注入攻击的鲁棒性，将攻击成功率（ASB）降低高达20%。这些结果表明，对指令层次结构的推理提供了一条通往可靠LLM的实用途径，其中对系统提示的更新会产生模型行为的可控且稳健的变化。



## **33. OpenLVLM-MIA: A Controlled Benchmark Revealing the Limits of Membership Inference Attacks on Large Vision-Language Models**

OpenLVLM-MIA：揭示大型视觉语言模型成员推断攻击局限性的受控基准 cs.CV

WACV2026 Accepted

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2510.16295v2) [paper-pdf](https://arxiv.org/pdf/2510.16295v2)

**Authors**: Ryoto Miyamoto, Xin Fan, Fuyuko Kido, Tsuneo Matsumoto, Hayato Yamana

**Abstract**: OpenLVLM-MIA is a new benchmark that highlights fundamental challenges in evaluating membership inference attacks (MIA) against large vision-language models (LVLMs). While prior work has reported high attack success rates, our analysis suggests that these results often arise from detecting distributional bias introduced during dataset construction rather than from identifying true membership status. To address this issue, we introduce a controlled benchmark of 6{,}000 images where the distributions of member and non-member samples are carefully balanced, and ground-truth membership labels are provided across three distinct training stages. Experiments using OpenLVLM-MIA demonstrated that the performance of state-of-the-art MIA methods approached chance-level. OpenLVLM-MIA, designed to be transparent and unbiased benchmark, clarifies certain limitations of MIA research on LVLMs and provides a solid foundation for developing stronger privacy-preserving techniques.

摘要: OpenLVLM-MIA是一个新基准，强调了评估针对大型视觉语言模型（LVLM）的成员资格推理攻击（MIA）的根本挑战。虽然之前的工作报告了很高的攻击成功率，但我们的分析表明，这些结果通常来自检测数据集构建期间引入的分布偏差，而不是识别真正的成员身份。为了解决这个问题，我们引入了一个由6{，}000张图像组成的受控基准，其中成员和非成员样本的分布经过仔细平衡，并在三个不同的训练阶段提供地面真相成员资格标签。使用OpenLVLM-MIA的实验表明，最先进的MIA方法的性能接近机会水平。OpenLVLM-MIA旨在成为透明和公正的基准，澄清了MIA对LVLM研究的某些局限性，并为开发更强大的隐私保护技术提供了坚实的基础。



## **34. Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness**

获得丰富或模具缩放：有利可图的交易推理计算的鲁棒性 cs.LG

21 pages

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2510.06790v2) [paper-pdf](https://arxiv.org/pdf/2510.06790v2)

**Authors**: Tavish McDonald, Bo Lei, Stanislav Fort, Bhavya Kailkhura, Brian Bartoldson

**Abstract**: Models are susceptible to adversarially out-of-distribution (OOD) data despite large training-compute investments into their robustification. Zaremba et al. (2025) make progress on this problem at test time, showing LLM reasoning improves satisfaction of model specifications designed to thwart attacks, resulting in a correlation between reasoning effort and robustness to jailbreaks. However, this benefit of test compute fades when attackers are given access to gradients or multimodal inputs. We address this gap, clarifying that inference-compute offers benefits even in such cases. Our approach argues that compositional generalization, through which OOD data is understandable via its in-distribution (ID) components, enables adherence to defensive specifications on adversarially OOD inputs. Namely, we posit the Robustness from Inference Compute Hypothesis (RICH): inference-compute defenses profit as the model's training data better reflects the attacked data's components. We empirically support this hypothesis across vision language model and attack types, finding robustness gains from test-time compute if specification following on OOD data is unlocked by compositional generalization. For example, InternVL 3.5 gpt-oss 20B gains little robustness when its test compute is scaled, but such scaling adds significant robustness if we first robustify its vision encoder. This correlation of inference-compute's robustness benefit with base model robustness is the rich-get-richer dynamic of the RICH: attacked data components are more ID for robustified models, aiding compositional generalization to OOD data. Thus, we advise layering train-time and test-time defenses to obtain their synergistic benefit.

摘要: 尽管模型的鲁棒性投入了大量的训练计算投资，但它们仍然容易受到不利的分布外（OOD）数据的影响。Zaremba等人（2025）在测试时在这个问题上取得了进展，表明LLM推理提高了旨在阻止攻击的模型规范的满意度，从而导致推理工作量和越狱稳健性之间的相关性。然而，当攻击者能够访问梯度或多模式输入时，测试计算的这种好处就会消失。我们解决了这一差距，澄清了即使在这种情况下，推理计算也能带来好处。我们的方法认为，组合概括（OOD数据可以通过其内分布（ID）组件来理解）使得能够遵守针对敌对OOD输入的防御规范。也就是说，我们从推理计算假设（RICH）中验证了鲁棒性：由于模型的训练数据更好地反映了受攻击数据的成分，推理计算防御会获利。我们在视觉语言模型和攻击类型中从经验上支持了这一假设，如果OOD数据上的规范通过组合概括解锁，则可以从测试时计算中找到鲁棒性的收益。例如，InternVL3.5 gtt-oss 20 B在扩展其测试计算时几乎没有获得鲁棒性，但如果我们首先对其视觉编码器进行鲁棒性验证，这种扩展会增加显着的鲁棒性。推理计算的稳健性优势与基础模型稳健性的这种相关性是RICH的富而富的动态：受攻击的数据组件对于稳健模型来说更具ID，有助于组合概括OOD数据。因此，我们建议将训练时和测试时防御分层，以获得协同效益。



## **35. SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations**

SECA：引发LLM幻觉的语义等效和一致攻击 cs.CL

Accepted at NeurIPS 2025. Code is available at https://github.com/Buyun-Liang/SECA

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2510.04398v2) [paper-pdf](https://arxiv.org/pdf/2510.04398v2)

**Authors**: Buyun Liang, Liangzu Peng, Jinqi Luo, Darshan Thaker, Kwan Ho Ryan Chan, René Vidal

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no semantic equivalence or semantic coherence errors compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at https://github.com/Buyun-Liang/SECA.

摘要: 大型语言模型（LLM）越来越多地部署在高风险领域。然而，最先进的LLM经常会产生幻觉，从而引发人们对其可靠性的严重担忧。之前的工作探索了LLM中幻觉引发的对抗攻击，但它经常会产生不切实际的提示，要么通过插入胡言乱语的标记，要么通过改变原来的含义。因此，这些方法对幻觉在实践中如何发生的了解有限。虽然计算机视觉中的对抗性攻击通常涉及对输入图像的现实修改，但寻找引发LLM幻觉的现实对抗性提示的问题在很大程度上仍未得到充分探索。为了解决这一差距，我们提出了语义等效和连贯的攻击（SECA）引起幻觉通过现实的修改提示，保留其意义，同时保持语义连贯性。我们的贡献有三重：（i）我们将为幻觉诱导寻找现实攻击制定为语义等价和一致性约束下的输入提示空间上的约束优化问题;（ii）我们引入约束保持零阶方法来有效地搜索对抗性但可行的提示;和（iii）我们通过开放式多-的实验来证明与现有方法相比，SECA的选择问题回答任务实现了更高的攻击成功率，同时几乎不会导致语义等效或语义一致性错误。SECA强调了开源和商业梯度不可访问的LLM对现实且合理的提示变化的敏感性。代码可在https://github.com/Buyun-Liang/SECA上获取。



## **36. When Ads Become Profiles: Uncovering the Invisible Risk of Web Advertising at Scale with LLMs**

当广告成为简介：利用LLM揭露大规模网络广告的隐形风险 cs.HC

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2509.18874v2) [paper-pdf](https://arxiv.org/pdf/2509.18874v2)

**Authors**: Baiyu Chen, Benjamin Tag, Hao Xue, Daniel Angus, Flora Salim

**Abstract**: Regulatory limits on explicit targeting have not eliminated algorithmic profiling on the Web, as optimisation systems still adapt ad delivery to users' private attributes. The widespread availability of powerful zero-shot multimodal Large Language Models (LLMs) has dramatically lowered the barrier for exploiting these latent signals for adversarial inference. We investigate this emerging societal risk, specifically how adversaries can now exploit these signals to reverse-engineer private attributes from ad exposure alone. We introduce a novel pipeline that leverages LLMs as adversarial inference engines to perform natural language profiling. Applying this method to a longitudinal dataset comprising over 435,000 ad impressions collected from 891 users, we conducted a large-scale study to assess the feasibility and precision of inferring private attributes from passive online ad observations. Our results demonstrate that off-the-shelf LLMs can accurately reconstruct complex user private attributes, including party preference, employment status, and education level, consistently outperforming strong census-based priors and matching or exceeding human social perception, while operating at only a fraction of the cost (223$\times$ lower) and time (52$\times$ faster) required by humans. Critically, actionable profiling is feasible even within short observation windows, indicating that prolonged tracking is not a prerequisite for a successful attack. These findings provide the first empirical evidence that ad streams serve as a high-fidelity digital footprint, enabling off-platform profiling that inherently bypasses current platform safeguards, highlighting a systemic vulnerability in the ad ecosystem and the urgent need for responsible web AI governance in the generative AI era. The code is available at https://github.com/Breezelled/when-ads-become-profiles.

摘要: 对显式定位的监管限制并没有消除网络上的算法分析，因为优化系统仍然根据用户的私人属性调整广告交付。强大的零镜头多模式大型语言模型（LLM）的广泛使用极大地降低了利用这些潜在信号进行对抗性推理的障碍。我们调查这种新出现的社会风险，特别是对手现在如何利用这些信号来仅从广告曝光中反向工程私人属性。我们引入了一种新颖的管道，利用LLM作为对抗推理引擎来执行自然语言分析。将这种方法应用于包含从891名用户收集的超过435，000个广告印象的纵向数据集，我们进行了一项大规模研究，以评估从被动在线广告观察中推断私人属性的可行性和精确性。我们的结果表明，现成的LLM可以准确地重建复杂的用户私人属性，包括政党偏好、就业状况和教育水平，始终优于基于人口普查的强大先验，并匹配或超过人类社会认知，同时运营成本仅为人类所需的一小部分（223美元\倍）和时间（52美元\倍）。至关重要的是，即使在短的观察窗口内，可操作的分析也是可行的，这表明长期跟踪并不是成功攻击的先决条件。这些发现提供了第一个经验证据，证明广告流可以充当高保真数字足迹，实现从本质上绕过当前平台保障措施的平台外分析，凸显了广告生态系统中的系统性漏洞以及生成性人工智能时代对负责任的网络人工智能治理的迫切需要。该代码可在https://github.com/Breezelled/when-ads-become-profiles上获取。



## **37. Membership Inference Attack against Large Language Model-based Recommendation Systems: A New Distillation-based Paradigm**

针对基于大型语言模型的推荐系统的成员推断攻击：一种基于蒸馏的新范式 cs.IR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2511.14763v2) [paper-pdf](https://arxiv.org/pdf/2511.14763v2)

**Authors**: Li Cuihong, Huang Xiaowen, Yin Chuanhuan, Sang Jitao

**Abstract**: Membership Inference Attack (MIA) aims to determine whether a specific data sample was included in the training dataset of a target model. Traditional MIA approaches rely on shadow models to mimic target model behavior, but their effectiveness diminishes for Large Language Model (LLM)-based recommendation systems due to the scale and complexity of training data. This paper introduces a novel knowledge distillation-based MIA paradigm tailored for LLM-based recommendation systems. Our method constructs a reference model via distillation, applying distinct strategies for member and non-member data to enhance discriminative capabilities. The paradigm extracts fused features (e.g., confidence, entropy, loss, and hidden layer vectors) from the reference model to train an attack model, overcoming limitations of individual features. Extensive experiments on extended datasets (Last.FM, MovieLens, Book-Crossing, Delicious) and diverse LLMs (T5, GPT-2, LLaMA3) demonstrate that our approach significantly outperforms shadow model-based MIAs and individual-feature baselines. The results show its practicality for privacy attacks in LLM-driven recommender systems.

摘要: 成员资格推理攻击（MIA）旨在确定特定数据样本是否包含在目标模型的训练数据集中。传统的MIA方法依赖影子模型来模仿目标模型行为，但由于训练数据的规模和复杂性，对于基于大型语言模型（LLM）的推荐系统，它们的有效性会降低。本文介绍了一种为基于LLM的推荐系统量身定制的新型基于知识蒸馏的MIA范式。我们的方法通过蒸馏构建参考模型，对成员和非成员数据应用不同的策略以增强区分能力。该范式提取融合的特征（例如，置信度、信息量、损失和隐藏层载体）从参考模型中训练攻击模型，克服单个特征的限制。对扩展数据集（Last.FM、MovieLens、Book-Crossing、Delicious）和各种LLM（T5、GPT-2、LLaMA 3）的广泛实验表明，我们的方法显着优于基于阴影模型的MIA和个体特征基线。结果表明了它对LLM驱动的推荐系统中的隐私攻击的实用性。



## **38. Involuntary Jailbreak**

非自愿越狱 cs.CR

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2508.13246v2) [paper-pdf](https://arxiv.org/pdf/2508.13246v2)

**Authors**: Yangyang Guo, Yangyan Li, Mohan Kankanhalli

**Abstract**: In this study, we disclose a worrying new vulnerability in Large Language Models (LLMs), which we term \textbf{involuntary jailbreak}. Unlike existing jailbreak attacks, this weakness is distinct in that it does not involve a specific attack objective, such as generating instructions for \textit{building a bomb}. Prior attack methods predominantly target localized components of the LLM guardrail. In contrast, involuntary jailbreaks may potentially compromise the entire guardrail structure, which our method reveals to be surprisingly fragile. We merely employ a single universal prompt to achieve this goal. In particular, we instruct LLMs to generate several questions that would typically be rejected, along with their corresponding in-depth responses (rather than a refusal). Remarkably, this simple prompt strategy consistently jailbreaks the majority of leading LLMs, including Claude Opus 4.1, Grok 4, Gemini 2.5 Pro, and GPT 4.1. We hope this problem can motivate researchers and practitioners to re-evaluate the robustness of LLM guardrails and contribute to stronger safety alignment in future.

摘要: 在这项研究中，我们揭示了大型语言模型（LLM）中一个令人担忧的新漏洞，我们将其称为\textBF{非自愿越狱}。与现有的越狱攻击不同，这个弱点的独特之处在于，它不涉及特定的攻击目标，例如为\texttit {building a bomb}生成指令。先前的攻击方法主要针对LLM护栏的局部部件。相比之下，非自愿越狱可能会损害整个护栏结构，而我们的方法表明该结构出奇地脆弱。我们只是使用一个普遍的提示来实现这一目标。特别是，我们指示LLM生成几个通常会被拒绝的问题，以及相应的深入回答（而不是拒绝）。值得注意的是，这种简单的提示策略持续破解了大多数领先的LLM，包括Claude Opus 4.1、Grok 4、Gemini 2.5 Pro和GPT 4.1。我们希望这个问题能够激励研究人员和从业者重新评估LLM护栏的稳健性，并为未来更强的安全性做出贡献。



## **39. Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference**

缓存中的影子：在LLM推理中揭示和减轻KV缓存的隐私风险 cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2508.09442v2) [paper-pdf](https://arxiv.org/pdf/2508.09442v2)

**Authors**: Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou, Yuke Hu, Chenxu Zhao, Zhihao Liu, Zhan Qin

**Abstract**: The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.

摘要: Key-Value（KV）缓存存储中间注意力计算（Key和Value对）以避免冗余计算，是加速大型语言模型（LLM）推理的基本机制。然而，这种效率优化引入了重大但未充分探索的隐私风险。本文首次对这些漏洞进行了全面分析，证明攻击者可以直接从KV缓存重建敏感用户输入。我们设计并实现了三种不同的攻击载体：直接倒置攻击、更广泛适用且更强大的碰撞攻击以及基于语义的注入攻击。这些方法证明了KV缓存隐私泄露问题的实用性和严重性。为了缓解这个问题，我们提出了KV-Cloak，这是一种新颖、轻量级且高效的防御机制。KV-Cloak使用基于可逆矩阵的混淆方案，结合操作符融合来保护KV-缓存。我们广泛的实验表明，KV-Cloak有效地阻止了所有提出的攻击，降低了随机噪音的重建质量。至关重要的是，它实现了这种强大的安全性，模型准确性几乎没有下降，性能负担最小，为值得信赖的LLM部署提供了实用的解决方案。



## **40. Large Language Models for Power System Security: A Novel Multi-Modal Approach for Anomaly Detection in Energy Management Systems**

电力系统安全的大型语言模型：能源管理系统异常检测的新型多模式方法 cs.CR

10 Figures; 6 Tables; Accepted, IEEE ACCESS 2025

**SubmitDate**: 2025-11-29    [abs](http://arxiv.org/abs/2508.10044v2) [paper-pdf](https://arxiv.org/pdf/2508.10044v2)

**Authors**: Aydin Zaboli, Junho Hong, Alexandru Stefanov, Chen-Ching Liu, Chul-Sang Hwang

**Abstract**: This paper elaborates on an extensive security framework specifically designed for energy management systems (EMSs), which effectively tackles the dynamic environment of cybersecurity vulnerabilities and/or system problems (SPs), accomplished through the incorporation of novel methodologies. A comprehensive multi-point attack/error model is initially proposed to systematically identify vulnerabilities throughout the entire EMS data processing pipeline, including post state estimation (SE) stealth attacks, EMS database manipulation, and human-machine interface (HMI) display corruption according to the real-time database (RTDB) storage. This framework acknowledges the interconnected nature of modern attack vectors, which utilize various phases of supervisory control and data acquisition (SCADA) data flow. Then, generative AI (GenAI)-based anomaly detection systems (ADSs) for EMSs are proposed for the first time in the power system domain to handle the scenarios. Further, a set-of-mark generative intelligence (SoM-GI) framework, which leverages multimodal analysis by integrating visual markers with rules considering the GenAI capabilities, is suggested to overcome inherent spatial reasoning limitations. The SoM-GI methodology employs systematic visual indicators to enable accurate interpretation of segmented HMI displays and detect visual anomalies that numerical methods fail to identify. Validation on the IEEE 14-Bus system shows the framework's effectiveness across scenarios, while visual analysis identifies inconsistencies. This integrated approach combines numerical analysis with visual pattern recognition and linguistic rules to protect against cyber threats and system errors.

摘要: 本文详细介绍了专门为能源管理系统（EMS）设计的广泛安全框架，该框架通过结合新颖的方法来有效地解决网络安全漏洞和/或系统问题（SP）的动态环境。最初提出了一个全面的多点攻击/错误模型，以系统地识别整个EMS数据处理管道中的漏洞，包括后状态估计（SE）隐形攻击、EMS数据库操纵和根据实时数据库（RTDB）存储的人机界面（HM）显示损坏。该框架承认现代攻击载体的相互关联性质，这些攻击载体利用监督控制和数据采集（DCS）数据流的各个阶段。然后，在电力系统领域首次提出了基于生成式人工智能（GenAI）的EMS异常检测系统（ADS）来处理这些场景。此外，还建议采用一种标记集生成智能（SoM-GI）框架，该框架通过将视觉标记与考虑GenAI能力的规则集成来利用多模式分析，以克服固有的空间推理限制。SoM-GI方法采用系统视觉指标来准确解释分段的人机界面显示并检测数字方法无法识别的视觉异常。在IEEE 14-Bus系统上的验证显示了该框架在各个场景中的有效性，而视觉分析则识别了不一致之处。这种集成方法将数字分析与视觉模式识别和语言规则相结合，以防止网络威胁和系统错误。



## **41. Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks**

视觉语言模型会泄露它们学到的东西吗？自适应令牌加权模型倒置攻击 cs.LG

Under review

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2508.04097v2) [paper-pdf](https://arxiv.org/pdf/2508.04097v2)

**Authors**: Ngoc-Bao Nguyen, Sy-Tuyen Ho, Koh Jun Hao, Ngai-Man Cheung

**Abstract**: Model inversion (MI) attacks pose significant privacy risks by reconstructing private training data from trained neural networks. While prior studies have primarily examined unimodal deep networks, the vulnerability of vision-language models (VLMs) remains largely unexplored. In this work, we present the first systematic study of MI attacks on VLMs to understand their susceptibility to leaking private visual training data. Our work makes two main contributions. First, tailored to the token-generative nature of VLMs, we introduce a suite of token-based and sequence-based model inversion strategies, providing a comprehensive analysis of VLMs' vulnerability under different attack formulations. Second, based on the observation that tokens vary in their visual grounding, and hence their gradients differ in informativeness for image reconstruction, we propose Sequence-based Model Inversion with Adaptive Token Weighting (SMI-AW) as a novel MI for VLMs. SMI-AW dynamically reweights each token's loss gradient according to its visual grounding, enabling the optimization to focus on visually informative tokens and more effectively guide the reconstruction of private images. Through extensive experiments and human evaluations on a range of state-of-the-art VLMs across multiple datasets, we show that VLMs are susceptible to training data leakage. Human evaluation of the reconstructed images yields an attack accuracy of 61.21%, underscoring the severity of these privacy risks. Notably, we demonstrate that publicly released VLMs are vulnerable to such attacks. Our study highlights the urgent need for privacy safeguards as VLMs become increasingly deployed in sensitive domains such as healthcare and finance. Additional experiments are provided in Supp.

摘要: 模型倒置（MI）攻击通过从训练的神经网络重建私人训练数据而构成重大隐私风险。虽然之前的研究主要研究了单模式深度网络，但视觉语言模型（VLM）的脆弱性在很大程度上仍然没有被探索。在这项工作中，我们首次对VLM的MI攻击进行了系统研究，以了解它们对泄露私人视觉训练数据的敏感性。我们的工作做出了两个主要贡献。首先，针对VLM的标记生成性质，我们引入了一套基于标记和基于序列的模型倒置策略，对不同攻击方案下的VLM脆弱性进行了全面分析。其次，根据标记的视觉基础不同，因此它们的梯度在图像重建的信息量上也不同的观察，我们提出了具有自适应标记加权的基于序列的模型倒置（SMI-AW）作为VLM的新型MI。SMI-AW根据每个代币的视觉基础动态重新加权其损失梯度，使优化能够专注于视觉信息丰富的代币，并更有效地指导私人图像的重建。通过对多个数据集中的一系列最先进的VLM进行广泛的实验和人类评估，我们表明VLM容易受到训练数据泄露的影响。对重建图像的人工评估产生了61.21%的攻击准确率，强调了这些隐私风险的严重性。值得注意的是，我们表明，公开发布的VLM容易受到这种攻击。我们的研究强调了隐私保护的迫切需要，因为VLM越来越多地部署在医疗保健和金融等敏感领域。补充中提供了其他实验。



## **42. Enhancing Jailbreak Attacks on LLMs via Persona Prompts**

通过女神异闻录加强对LLM的越狱攻击 cs.CR

Workshop on LLM Persona Modeling at NeurIPS 2025

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2507.22171v2) [paper-pdf](https://arxiv.org/pdf/2507.22171v2)

**Authors**: Zheng Zhang, Peilin Zhao, Deheng Ye, Hao Wang

**Abstract**: Jailbreak attacks aim to exploit large language models (LLMs) by inducing them to generate harmful content, thereby revealing their vulnerabilities. Understanding and addressing these attacks is crucial for advancing the field of LLM safety. Previous jailbreak approaches have mainly focused on direct manipulations of harmful intent, with limited attention to the impact of persona prompts. In this study, we systematically explore the efficacy of persona prompts in compromising LLM defenses. We propose a genetic algorithm-based method that automatically crafts persona prompts to bypass LLM's safety mechanisms. Our experiments reveal that: (1) our evolved persona prompts reduce refusal rates by 50-70% across multiple LLMs, and (2) these prompts demonstrate synergistic effects when combined with existing attack methods, increasing success rates by 10-20%. Our code and data are available at https://github.com/CjangCjengh/Generic_Persona.

摘要: 越狱攻击旨在通过诱导大型语言模型（LLM）生成有害内容来利用它们，从而揭示它们的漏洞。理解和解决这些攻击对于推进LLM安全领域至关重要。以前的越狱方法主要集中在对有害意图的直接操纵上，对人物角色提示的影响关注有限。在这项研究中，我们系统地探讨了影响LLM防御的角色提示的功效。我们提出了一种基于遗传算法的方法，可以自动制作角色提示以绕过LLM的安全机制。我们的实验表明：（1）我们进化的角色提示将多个LLM的拒绝率降低50-70%，并且（2）这些提示与现有的攻击方法结合时表现出协同效应，将成功率提高10- 20%。我们的代码和数据可在https://github.com/CjangCjengh/Generic_Persona上获取。



## **43. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

论LLM在对抗性攻击中言语信心的稳健性 cs.CL

Published in NeurIPS 2025

**SubmitDate**: 2025-11-28    [abs](http://arxiv.org/abs/2507.06489v2) [paper-pdf](https://arxiv.org/pdf/2507.06489v2)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.

摘要: 大型语言模型（LLM）产生的强大言语信心对于LLM的部署至关重要，以帮助确保许多应用程序（包括涉及人机交互的应用程序）的透明度、信任和安全性。在本文中，我们首次对对抗攻击下言语信心的稳健性进行了全面研究。我们通过干扰和基于越狱的方法引入了针对言语信心分数的攻击框架，并证明这些攻击会显着损害言语信心估计并导致答案频繁变化。我们检查了各种提示策略、模型大小和应用领域，揭示了当前的言语自信很脆弱，并且常用的防御技术在很大程度上无效或适得其反。我们的研究结果强调了为LLM中的信心表达设计稳健的机制的必要性，因为即使是微妙的语义保留修改也可能导致反应中的误导性信心。



## **44. SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism**

SafeTLR：通过删除然后恢复机制在多模式LLM中进行令牌级越狱防御 cs.CR

Accepted by NeurIPS 2025

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2507.01513v2) [paper-pdf](https://arxiv.org/pdf/2507.01513v2)

**Authors**: Beitao Chen, Xinyu Lyu, Lianli Gao, Jingkuan Song, Heng Tao Shen

**Abstract**: By incorporating visual inputs, Multimodal Large Language Models (MLLMs) extend LLMs to support visual reasoning. However, this integration also introduces new vulnerabilities, making MLLMs susceptible to multimodal jailbreak attacks and hindering their safe deployment.Existing defense methods, including Image-to-Text Translation, Safe Prompting, and Multimodal Safety Tuning, attempt to address this by aligning multimodal inputs with LLMs' built-in safeguards.Yet, they fall short in uncovering root causes of multimodal vulnerabilities, particularly how harmful multimodal tokens trigger jailbreak in MLLMs? Consequently, they remain vulnerable to text-driven multimodal jailbreaks, often exhibiting overdefensive behaviors and imposing heavy training overhead.To bridge this gap, we present an comprehensive analysis of where, how and which harmful multimodal tokens bypass safeguards in MLLMs. Surprisingly, we find that less than 1% tokens in early-middle layers are responsible for inducing unsafe behaviors, highlighting the potential of precisely removing a small subset of harmful tokens, without requiring safety tuning, can still effectively improve safety against jailbreaks. Motivated by this, we propose Safe Prune-then-Restore (SafePTR), an training-free defense framework that selectively prunes harmful tokens at vulnerable layers while restoring benign features at subsequent layers.Without incurring additional computational overhead, SafePTR significantly enhances the safety of MLLMs while preserving efficiency. Extensive evaluations across three MLLMs and five benchmarks demonstrate SafePTR's state-of-the-art performance in mitigating jailbreak risks without compromising utility.

摘要: 通过结合视觉输入，多模式大型语言模型（MLLM）扩展了LLM以支持视觉推理。然而，这种集成也引入了新的漏洞，使MLLM容易受到多模式越狱攻击并阻碍其安全部署。现有的防御方法，包括图像到文本翻译、安全预算处理和多模式安全调优，试图通过将多模式输入与LLM的内置保护措施相一致来解决这个问题。然而，它们未能发现多模式漏洞的根本原因，特别是有害的多模式代币如何触发MLLM越狱？因此，他们仍然容易受到文本驱动的多模式越狱的影响，通常表现出过度防御行为并施加沉重的培训费用。为了弥合这一差距，我们对MLLM中的哪些有害多模式代币在哪里、如何以及哪些方式绕过保障措施进行了全面分析。令人惊讶的是，我们发现早期中层中只有不到1%的代币会引发不安全行为，这凸显了在不需要安全调整的情况下精确删除有害代币的一小部分的潜力，仍然可以有效地提高针对越狱的安全性。出于此动机，我们提出了Safe Prune-then-Restore（SafeTLR），这是一种免训练的防御框架，可以选择性地修剪脆弱层的有害令牌，同时在后续层恢复良性特征。在不产生额外计算负担的情况下，SafeTLR显着增强了MLLM的安全性，同时保持了效率。对三个MLLM和五个基准的广泛评估表明，SafeTLR在缓解越狱风险而不影响实用性方面具有最先进的性能。



## **45. QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA**

QA-LIGN：通过宪法分解的QA调整LLM cs.CL

Findings of the Association for Computational Linguistics: EMNLP 2025, pages 20619-20642, Suzhou, China

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2506.08123v5) [paper-pdf](https://arxiv.org/pdf/2506.08123v5)

**Authors**: Jacob Dineen, Aswin RRV, Qin Liu, Zhikun Xu, Xiao Ye, Ming Shen, Zhaonan Li, Shijie Lu, Chitta Baral, Muhao Chen, Ben Zhou

**Abstract**: Alignment of large language models (LLMs) with principles like helpfulness, honesty, and harmlessness typically relies on scalar rewards that obscure which objectives drive the training signal. We introduce QA-LIGN, which decomposes monolithic rewards into interpretable principle-specific evaluations through structured natural language programs. Models learn through a draft, critique, and revise pipeline, where symbolic evaluation against the rubrics provides transparent feedback for both initial and revised responses during GRPO training. Applied to uncensored Llama-3.1-8B-Instruct, QA-LIGN reduces attack success rates by up to 68.7% while maintaining a 0.67% false refusal rate, achieving Pareto optimal safety-helpfulness performance and outperforming both DPO and GRPO with state-of-the-art reward models given equivalent training. These results demonstrate that making reward signals interpretable and modular improves alignment effectiveness, suggesting transparency enhances LLM safety.

摘要: 大型语言模型（LLM）与乐于助人、诚实和无害等原则的一致通常依赖于量化奖励，这些奖励模糊了哪些目标驱动训练信号。我们引入QA-LIGN，它通过结构化自然语言程序将单一奖励分解为可解释的特定于原则的评估。模型通过起草、评论和修改管道进行学习，其中针对主题的象征性评估为GRPO培训期间的初始和修改响应提供透明的反馈。应用于未经审查的Llama-3.1- 8B-Direct，QA-LIGN将攻击成功率降低高达68.7%，同时保持0.67%的错误拒绝率，实现了帕累托最佳安全帮助性能，并在同等培训的情况下优于DPO和GRPO。这些结果表明，使奖励信号可解释和模块化可以提高对齐有效性，这表明透明度增强了LLM的安全性。



## **46. Guard Me If You Know Me: Protecting Specific Face-Identity from Deepfakes**

如果你认识我，请保护我：保护特定面部身份免受Deepfakes的侵害 cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2505.19582v2) [paper-pdf](https://arxiv.org/pdf/2505.19582v2)

**Authors**: Kaiqing Lin, Zhiyuan Yan, Ke-Yue Zhang, Li Hao, Yue Zhou, Yuzhen Lin, Weixiang Li, Taiping Yao, Shouhong Ding, Bin Li

**Abstract**: Securing personal identity against deepfake attacks is increasingly critical in the digital age, especially for celebrities and political figures whose faces are easily accessible and frequently targeted. Most existing deepfake detection methods focus on general-purpose scenarios and often ignore the valuable prior knowledge of known facial identities, e.g., "VIP individuals" whose authentic facial data are already available. In this paper, we propose \textbf{VIPGuard}, a unified multimodal framework designed to capture fine-grained and comprehensive facial representations of a given identity, compare them against potentially fake or similar-looking faces, and reason over these comparisons to make accurate and explainable predictions. Specifically, our framework consists of three main stages. First, fine-tune a multimodal large language model (MLLM) to learn detailed and structural facial attributes. Second, we perform identity-level discriminative learning to enable the model to distinguish subtle differences between highly similar faces, including real and fake variations. Finally, we introduce user-specific customization, where we model the unique characteristics of the target face identity and perform semantic reasoning via MLLM to enable personalized and explainable deepfake detection. Our framework shows clear advantages over previous detection works, where traditional detectors mainly rely on low-level visual cues and provide no human-understandable explanations, while other MLLM-based models often lack a detailed understanding of specific face identities. To facilitate the evaluation of our method, we built a comprehensive identity-aware benchmark called \textbf{VIPBench} for personalized deepfake detection, involving the latest 7 face-swapping and 7 entire face synthesis techniques for generation. The code is available at https://github.com/KQL11/VIPGuard .

摘要: 在数字时代，保护个人身份免受Deepfake攻击变得越来越重要，尤其是对于面部易于接触且经常成为攻击目标的名人和政治人物。大多数现有的Deepfake检测方法都专注于通用场景，并且经常忽略已知面部身份的宝贵先验知识，例如，其真实面部数据已经可用的“VIP个人”。在本文中，我们提出了\textBF{VIPGuard}，这是一个统一的多模式框架，旨在捕获给定身份的细粒度和全面的面部表示，将它们与潜在的虚假或相似的面部进行比较，并推理这些比较以做出准确且可解释的预测。具体来说，我们的框架由三个主要阶段组成。首先，微调多模式大型语言模型（MLLM）以学习详细和结构化的面部属性。其次，我们执行身份级别的辨别学习，使模型能够区分高度相似的面孔之间的细微差异，包括真实和虚假的变体。最后，我们引入了特定于用户的定制，其中我们对目标人脸身份的独特特征进行建模，并通过MLLM执行语义推理，以实现个性化和可解释的深度伪造检测。与之前的检测工作相比，我们的框架显示出明显的优势，传统的检测器主要依赖于低级视觉线索，并且不提供人类可理解的解释，而其他基于MLLM的模型通常缺乏对特定面部身份的详细了解。为了促进对我们的方法的评估，我们构建了一个名为\textBF{VIPBench}的全面身份感知基准，用于个性化深度伪造检测，其中涉及最新的7种面部交换和7种完整面部合成技术。该代码可在https://github.com/KQL11/VIPGuard上获取。



## **47. Les Dissonances: Cross-Tool Harvesting and Polluting in Pool-of-Tools Empowered LLM Agents**

Les Dissonance：工具池中的跨工具收获和污染赋予LLM代理人权力 cs.CR

Network and Distributed System Security (NDSS) Symposium 2026

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2504.03111v3) [paper-pdf](https://arxiv.org/pdf/2504.03111v3)

**Authors**: Zichuan Li, Jian Cui, Xiaojing Liao, Luyi Xing

**Abstract**: Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 66 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 75% are vulnerable to XTHP attacks, highlighting the prevalence of this threat.

摘要: 大型语言模型（LLM）代理是由LLM支持的自治系统，能够通过利用一组工具进行推理和规划来解决问题。然而，LLM代理中多工具功能的集成在安全管理工具、确保其兼容性、处理依赖关系以及保护LLM代理工作流程中的控制流方面带来了挑战。本文中，我们首次对支持多工具的LLM代理中的任务控制流进行了系统性安全分析。我们识别了一种新型威胁，即跨工具收获和污染（XTHP），它包括多个攻击载体，首先劫持代理任务的正常控制流，然后收集和污染LLM代理系统内的机密或私人信息。为了了解这种威胁的影响，我们开发了Chord，这是一种动态扫描工具，旨在自动检测容易受到XTHP攻击的现实世界代理工具。我们对两个主要LLM代理开发框架LangChain和LlamaIndex存储库中的66个现实工具进行了评估，发现了一个重大的安全问题：75%容易受到XTHP攻击，凸显了这种威胁的普遍性。



## **48. AED: Automatic Discovery of Effective and Diverse Vulnerabilities for Autonomous Driving Policy with Large Language Models**

AED：利用大型语言模型自动发现自动驾驶政策的有效且多样化的漏洞 cs.CR

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2503.20804v2) [paper-pdf](https://arxiv.org/pdf/2503.20804v2)

**Authors**: Le Qiu, Zelai Xu, Qixin Tan, Wenhao Tang, Chao Yu, Yu Wang

**Abstract**: Assessing the safety of autonomous driving policy is of great importance, and reinforcement learning (RL) has emerged as a powerful method for discovering critical vulnerabilities in driving policies. However, existing RL-based approaches often struggle to identify vulnerabilities that are both effective-meaning the autonomous vehicle is genuinely responsible for the accidents-and diverse-meaning they span various failure types. To address these challenges, we propose AED, a framework that uses large language models (LLMs) to automatically discover effective and diverse vulnerabilities in autonomous driving policies. We first utilize an LLM to automatically design reward functions for RL training. Then we let the LLM consider a diverse set of accident types and train adversarial policies for different accident types in parallel. Finally, we use preference-based learning to filter ineffective accidents and enhance the effectiveness of each vulnerability. Experiments across multiple simulated traffic scenarios and tested policies show that AED uncovers a broader range of vulnerabilities and achieves higher attack success rates compared with expert-designed rewards, thereby reducing the need for manual reward engineering and improving the diversity and effectiveness of vulnerability discovery. The implementation can be found on: https://github.com/thu-nics/AED .

摘要: 评估自动驾驶策略的安全性非常重要，强化学习（RL）已成为发现驾驶策略中关键漏洞的强大方法。然而，现有的基于RL的方法通常很难识别既有效的漏洞（这意味着自动驾驶汽车真正对事故负责）又多样化的漏洞（这意味着它们跨越各种故障类型）。为了应对这些挑战，我们提出AED，这是一个使用大型语言模型（LLM）自动发现自动驾驶政策中有效且多样化的漏洞的框架。我们首先利用LLM来自动设计RL培训的奖励函数。然后，我们让LLM考虑一系列不同的事故类型，并并行训练不同事故类型的对抗政策。最后，我们使用基于偏好的学习来过滤无效事故并增强每个漏洞的有效性。跨多个模拟流量场景和测试策略的实验表明，与专家设计的奖励相比，AED发现了更广泛的漏洞，并实现了更高的攻击成功率，从而减少了手动奖励工程的需求，提高了漏洞发现的多样性和有效性。该实现可在https://github.com/thu-nics/AED上找到。



## **49. Efficient LLM-Jailbreaking via Multimodal-LLM Jailbreak**

通过Multimodal-LLM越狱实现高效的LLM越狱 cs.AI

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2405.20015v3) [paper-pdf](https://arxiv.org/pdf/2405.20015v3)

**Authors**: Haoxuan Ji, Zheng Lin, Zhenxing Niu, Xinbo Gao, Gang Hua

**Abstract**: This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreak methods that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) built upon the target LLM. Subsequently, we perform an efficient MLLM jailbreak and obtain a jailbreaking embedding. Finally, we convert the embedding into a textual jailbreaking suffix to carry out the jailbreak of target LLM. Compared to the direct LLM-jailbreak methods, our indirect jailbreaking approach is more efficient, as MLLMs are more vulnerable to jailbreak than pure LLM. Additionally, to improve the attack success rate of jailbreak, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art jailbreak methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class generalization abilities.

摘要: 本文重点关注针对大型语言模型（LLM）的越狱攻击，促使它们生成令人反感的内容来响应有害的用户查询。与之前直接面向LLM的LLM越狱方法不同，我们的方法首先构建基于目标LLM的多模式大型语言模型（MLLM）。随后，我们执行高效的MLLM越狱并获得越狱嵌入。最后，我们将嵌入转换为文本越狱后缀来执行目标LLM的越狱。与直接LLM越狱方法相比，我们的间接越狱方法更有效，因为MLLM比纯粹的LLM更容易受到越狱的影响。此外，为了提高越狱的攻击成功率，我们提出了一种图像-文本语义匹配方案来识别合适的初始输入。大量实验表明，我们的方法在效率和有效性方面都超过了当前最先进的越狱方法。此外，我们的方法具有出色的跨类概括能力。



## **50. Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation**

撬锁LLM：使用代币级操纵的基于日志的越狱 cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2405.13068v3) [paper-pdf](https://arxiv.org/pdf/2405.13068v3)

**Authors**: Yuxi Li, Yi Liu, Yuekang Li, Ling Shi, Gelei Deng, Shengquan Chen, Kailong Wang

**Abstract**: Large language models (LLMs) have transformed the field of natural language processing, but they remain susceptible to jailbreaking attacks that exploit their capabilities to generate unintended and potentially harmful content. Existing token-level jailbreaking techniques, while effective, face scalability and efficiency challenges, especially as models undergo frequent updates and incorporate advanced defensive measures. In this paper, we introduce JailMine, an innovative token-level manipulation approach that addresses these limitations effectively. JailMine employs an automated "mining" process to elicit malicious responses from LLMs by strategically selecting affirmative outputs and iteratively reducing the likelihood of rejection. Through rigorous testing across multiple well-known LLMs and datasets, we demonstrate JailMine's effectiveness and efficiency, achieving a significant average reduction of 86% in time consumed while maintaining high success rates averaging 95%, even in the face of evolving defensive strategies. Our work contributes to the ongoing effort to assess and mitigate the vulnerability of LLMs to jailbreaking attacks, underscoring the importance of continued vigilance and proactive measures to enhance the security and reliability of these powerful language models.

摘要: 大型语言模型（LLM）已经改变了自然语言处理领域，但它们仍然容易受到越狱攻击，这些攻击利用它们的能力来生成意想不到的和潜在有害的内容。现有的代币级越狱技术虽然有效，但面临可扩展性和效率的挑战，特别是当模型经历频繁更新并纳入先进的防御措施时。在本文中，我们介绍了JailMine，这是一种创新的代币级操纵方法，可以有效地解决这些限制。JailMine采用自动化“挖掘”流程，通过战略性地选择肯定输出并迭代降低拒绝的可能性来引发LLM的恶意响应。通过对多个知名LLM和数据集的严格测试，我们证明了JailMine的有效性和效率，实现了平均86%的时间大幅减少，同时保持了平均95%的高成功率，即使面对不断变化的防御策略。我们的工作有助于评估和减轻LLM对越狱攻击的脆弱性的持续努力，强调了持续保持警惕和采取积极主动措施以增强这些强大语言模型的安全性和可靠性的重要性。



