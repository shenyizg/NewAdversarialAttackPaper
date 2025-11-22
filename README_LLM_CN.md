# Latest Large Language Model Attack Papers
**update at 2025-11-22 11:17:02**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Large Language Model-Based Reward Design for Deep Reinforcement Learning-Driven Autonomous Cyber Defense**

基于大语言模型的深度强化学习驱动的自主网络防御奖励设计 cs.LG

Accepted in the AAAI-26 Workshop on Artificial Intelligence for Cyber Security (AICS)

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16483v1) [paper-pdf](https://arxiv.org/pdf/2511.16483v1)

**Authors**: Sayak Mukherjee, Samrat Chatterjee, Emilie Purvine, Ted Fujimoto, Tegan Emerson

**Abstract**: Designing rewards for autonomous cyber attack and defense learning agents in a complex, dynamic environment is a challenging task for subject matter experts. We propose a large language model (LLM)-based reward design approach to generate autonomous cyber defense policies in a deep reinforcement learning (DRL)-driven experimental simulation environment. Multiple attack and defense agent personas were crafted, reflecting heterogeneity in agent actions, to generate LLM-guided reward designs where the LLM was first provided with contextual cyber simulation environment information. These reward structures were then utilized within a DRL-driven attack-defense simulation environment to learn an ensemble of cyber defense policies. Our results suggest that LLM-guided reward designs can lead to effective defense strategies against diverse adversarial behaviors.

摘要: 对于主题专家来说，在复杂、动态的环境中为自主网络攻击和防御学习代理设计奖励是一项具有挑战性的任务。我们提出了一种基于大语言模型（LLM）的奖励设计方法，以在深度强化学习（DRL）驱动的实验模拟环境中生成自主网络防御策略。精心设计了多个攻击和防御代理角色，反映了代理动作的多样性，以生成LLM引导的奖励设计，其中LLM首先被提供上下文网络模拟环境信息。然后在DRL驱动的攻击防御模拟环境中使用这些奖励结构来学习一整套网络防御策略。我们的结果表明，LLM指导的奖励设计可以制定针对不同对抗行为的有效防御策略。



## **2. Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security**

Q-MLLM：鲁棒多模式大型语言模型安全性的载体量化 cs.CR

Accepted by NDSS 2026

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16229v1) [paper-pdf](https://arxiv.org/pdf/2511.16229v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in cross-modal understanding, but remain vulnerable to adversarial attacks through visual inputs despite robust textual safety mechanisms. These vulnerabilities arise from two core weaknesses: the continuous nature of visual representations, which allows for gradient-based attacks, and the inadequate transfer of text-based safety mechanisms to visual content. We introduce Q-MLLM, a novel architecture that integrates two-level vector quantization to create a discrete bottleneck against adversarial attacks while preserving multimodal reasoning capabilities. By discretizing visual representations at both pixel-patch and semantic levels, Q-MLLM blocks attack pathways and bridges the cross-modal safety alignment gap. Our two-stage training methodology ensures robust learning while maintaining model utility. Experiments demonstrate that Q-MLLM achieves significantly better defense success rate against both jailbreak attacks and toxic image attacks than existing approaches. Notably, Q-MLLM achieves perfect defense success rate (100\%) against jailbreak attacks except in one arguable case, while maintaining competitive performance on multiple utility benchmarks with minimal inference overhead. This work establishes vector quantization as an effective defense mechanism for secure multimodal AI systems without requiring expensive safety-specific fine-tuning or detection overhead. Code is available at https://github.com/Amadeuszhao/QMLLM.

摘要: 多模式大型语言模型（MLLM）在跨模式理解方面表现出了令人印象深刻的能力，但尽管具有强大的文本安全机制，但仍然容易受到视觉输入的对抗攻击。这些漏洞源于两个核心弱点：视觉表示的连续性（允许基于梯度的攻击）以及基于文本的安全机制向视觉内容的不充分转移。我们引入了Q-MLLM，这是一种新颖的架构，它集成了两级量化，以创建针对对抗性攻击的离散瓶颈，同时保留多模式推理能力。通过在像素补丁和语义层面离散化视觉表示，Q-MLLM阻止攻击途径并弥合跨模式安全对齐差距。我们的两阶段训练方法确保稳健的学习，同时保持模型效用。实验表明，与现有方法相比，Q-MLLM在针对越狱攻击和有毒图像攻击的防御成功率明显更高。值得注意的是，Q-MLLM在针对越狱攻击时实现了完美的防御成功率（100%），但在一种可发现的情况下，同时以最小的推理费用在多个实用工具基准上保持竞争性能。这项工作将载体量化建立为安全多模式人工智能系统的有效防御机制，而不需要昂贵的安全特定微调或检测费用。代码可在https://github.com/Amadeuszhao/QMLLM上获取。



## **3. PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization**

PSM：通过LLM引导的黑盒优化实现快速灵敏度最小化 cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16209v1) [paper-pdf](https://arxiv.org/pdf/2511.16209v1)

**Authors**: Huseein Jawad, Nicolas Brunel

**Abstract**: System prompts are critical for guiding the behavior of Large Language Models (LLMs), yet they often contain proprietary logic or sensitive information, making them a prime target for extraction attacks. Adversarial queries can successfully elicit these hidden instructions, posing significant security and privacy risks. Existing defense mechanisms frequently rely on heuristics, incur substantial computational overhead, or are inapplicable to models accessed via black-box APIs. This paper introduces a novel framework for hardening system prompts through shield appending, a lightweight approach that adds a protective textual layer to the original prompt. Our core contribution is the formalization of prompt hardening as a utility-constrained optimization problem. We leverage an LLM-as-optimizer to search the space of possible SHIELDs, seeking to minimize a leakage metric derived from a suite of adversarial attacks, while simultaneously preserving task utility above a specified threshold, measured by semantic fidelity to baseline outputs. This black-box, optimization-driven methodology is lightweight and practical, requiring only API access to the target and optimizer LLMs. We demonstrate empirically that our optimized SHIELDs significantly reduce prompt leakage against a comprehensive set of extraction attacks, outperforming established baseline defenses without compromising the model's intended functionality. Our work presents a paradigm for developing robust, utility-aware defenses in the escalating landscape of LLM security. The code is made public on the following link: https://github.com/psm-defense/psm

摘要: 系统提示对于指导大型语言模型（LLM）的行为至关重要，但它们通常包含专有逻辑或敏感信息，使其成为提取攻击的主要目标。对抗性查询可以成功地引出这些隐藏指令，从而构成重大的安全和隐私风险。现有的防御机制通常依赖于启发式方法，会产生大量的计算负担，或者不适用于通过黑匣子API访问的模型。本文介绍了一种通过屏蔽附加来强化系统提示的新颖框架，这是一种轻量级方法，可以在原始提示中添加保护性文本层。我们的核心贡献是将即时硬化形式化为一个受效用约束的优化问题。我们利用LLM作为优化器来搜索可能的SHIELD的空间，寻求最大限度地减少从一系列对抗攻击中获得的泄漏指标，同时将任务效用保持在指定阈值以上，该阈值通过基线输出的语义保真度来衡量。这种黑匣子、优化驱动的方法是轻量级且实用的，仅需要API访问目标和优化器LLM。我们通过经验证明，我们优化的SHIELD显着减少了针对一系列全面提取攻击的即时泄漏，在不损害模型预期功能的情况下优于既定的基线防御。我们的工作提供了一个在LLM安全不断升级的环境中开发强大的、实用程序感知的防御的范式。该代码在以下链接上公开：https://github.com/psm-defense/psm



## **4. When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models**

当对齐失败时：对视觉-语言-动作模型的多模式对抗攻击 cs.CV

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16203v1) [paper-pdf](https://arxiv.org/pdf/2511.16203v1)

**Authors**: Yuping Yan, Yuhan Xie, Yinxin Zhang, Lingjuan Lyu, Yaochu Jin

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.

摘要: 视觉-语言-动作模型（VLA）最近在具体环境中取得了显着进展，使机器人能够通过统一的多模式理解来感知、推理和行动。尽管它们的能力令人印象深刻，但这些系统的对抗鲁棒性在很大程度上仍未得到探索，尤其是在现实的多模式和黑匣子条件下。现有的研究主要关注单模式扰动，而忽视了从根本上影响体现推理和决策的跨模式失调。本文介绍了VLA-Fool，这是对白盒和黑盒设置下具体VLA模型中多模式对抗鲁棒性的全面研究。VLA-Fool统一了三个级别的多模式对抗攻击：（1）通过基于梯度和基于预算的操纵进行文本扰动，（2）通过补丁和噪音失真进行视觉扰动，以及（3）故意破坏感知和指令之间的语义对应性的跨模式失准攻击。我们进一步将VLA感知的语义空间融入到语言提示中，开发了第一个自动制作和语义引导的提示框架。使用微调的OpenVLA模型对LIBERO基准进行的实验表明，即使是微小的多峰扰动也会导致显着的行为偏差，这表明了体现多峰对齐的脆弱性。



## **5. What Your Features Reveal: Data-Efficient Black-Box Feature Inversion Attack for Split DNNs**

您的功能揭示了什么：针对拆分DNN的数据高效黑匣子功能倒置攻击 cs.CV

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15316v1) [paper-pdf](https://arxiv.org/pdf/2511.15316v1)

**Authors**: Zhihan Ren, Lijun He, Jiaxi Liang, Xinzhu Fu, Haixia Bi, Fan Li

**Abstract**: Split DNNs enable edge devices by offloading intensive computation to a cloud server, but this paradigm exposes privacy vulnerabilities, as the intermediate features can be exploited to reconstruct the private inputs via Feature Inversion Attack (FIA). Existing FIA methods often produce limited reconstruction quality, making it difficult to assess the true extent of privacy leakage. To reveal the privacy risk of the leaked features, we introduce FIA-Flow, a black-box FIA framework that achieves high-fidelity image reconstruction from intermediate features. To exploit the semantic information within intermediate features, we design a Latent Feature Space Alignment Module (LFSAM) to bridge the semantic gap between the intermediate feature space and the latent space. Furthermore, to rectify distributional mismatch, we develop Deterministic Inversion Flow Matching (DIFM), which projects off-manifold features onto the target manifold with one-step inference. This decoupled design simplifies learning and enables effective training with few image-feature pairs. To quantify privacy leakage from a human perspective, we also propose two metrics based on a large vision-language model. Experiments show that FIA-Flow achieves more faithful and semantically aligned feature inversion across various models (AlexNet, ResNet, Swin Transformer, DINO, and YOLO11) and layers, revealing a more severe privacy threat in Split DNNs than previously recognized.

摘要: 拆分DNN通过将密集计算卸载到云服务器来支持边缘设备，但这种范式暴露了隐私漏洞，因为可以利用中间功能通过特征倒置攻击（FIA）重建私人输入。现有的FIA方法通常产生有限的重建质量，因此很难评估隐私泄露的真实程度。为了揭示泄露特征的隐私风险，我们引入了FIA-Flow，这是一种黑匣子FIA框架，可以从中间特征实现高保真图像重建。为了利用中间特征中的语义信息，我们设计了一个潜在特征空间对齐模块（LFSam）来弥合中间特征空间和潜在空间之间的语义差距。此外，为了纠正分布不匹配，我们开发了确定性反演流匹配（DIFM），该方法通过一步推理将流形外特征投影到目标流形上。这种解耦的设计简化了学习，并且能够使用很少的图像特征对进行有效的训练。为了从人类的角度量化隐私泄露，我们还提出了两个基于大型视觉语言模型的指标。实验表明，FIA-Flow在各种模型（AlexNet、ResNet、Swin Transformer、DINO和YOLO 11）和层中实现了更忠实、语义一致的特征倒置，揭示了Split DNN中比之前认识到的更严重的隐私威胁。



## **6. Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models**

对抗性诗歌作为大型语言模型中通用的单轮越狱机制 cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15304v2) [paper-pdf](https://arxiv.org/pdf/2511.15304v2)

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.

摘要: 我们提供的证据表明，对抗性诗歌可以作为大型语言模型（LLM）的通用单轮越狱技术。在25个前沿专有和开放重量模型中，精心策划的诗意提示产生了很高的攻击成功率（ASB），一些提供商超过了90%。MLCommons和EU CoP风险分类的映射提示表明，诗意攻击跨CBRN、操纵、网络犯罪和失去控制领域转移。通过标准化元提示将1，200个MLCommons有害提示转换为诗句，产生的ASB比散文基线高出18倍。使用3名开放权重LLM评委的整体评估输出，他们的二元安全性评估在分层的人类标记子集上进行了验证。诗意框架的平均越狱成功率为62%，元提示转换的平均越狱成功率约为43%（与非诗意基线相比），大大优于非诗意基线，并揭示了示范家庭和安全培训方法之间的系统性弱点。这些研究结果表明，仅靠风格差异就可以规避当代安全机制，这表明当前对齐方法和评估协议存在根本性局限性。



## **7. Securing AI Agents Against Prompt Injection Attacks**

保护人工智能代理免受即时注入攻击 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15759v1) [paper-pdf](https://arxiv.org/pdf/2511.15759v1)

**Authors**: Badrinath Ramakrishnan, Akshaya Balaji

**Abstract**: Retrieval-augmented generation (RAG) systems have become widely used for enhancing large language model capabilities, but they introduce significant security vulnerabilities through prompt injection attacks. We present a comprehensive benchmark for evaluating prompt injection risks in RAG-enabled AI agents and propose a multi-layered defense framework. Our benchmark includes 847 adversarial test cases across five attack categories: direct injection, context manipulation, instruction override, data exfiltration, and cross-context contamination. We evaluate three defense mechanisms: content filtering with embedding-based anomaly detection, hierarchical system prompt guardrails, and multi-stage response verification, across seven state-of-the-art language models. Our combined framework reduces successful attack rates from 73.2% to 8.7% while maintaining 94.3% of baseline task performance. We release our benchmark dataset and defense implementation to support future research in AI agent security.

摘要: 检索增强生成（RAG）系统已被广泛用于增强大型语言模型能力，但它们通过提示注入攻击引入了严重的安全漏洞。我们提出了一个全面的基准来评估支持RAG的人工智能代理中的即时注入风险，并提出了一个多层防御框架。我们的基准测试包括跨越五种攻击类别的847个对抗测试案例：直接注入、上下文操纵、指令覆盖、数据溢出和跨上下文污染。我们评估了三种防御机制：基于嵌入的异常检测的内容过滤、分层系统提示护栏和跨七种最先进语言模型的多阶段响应验证。我们的组合框架将成功攻击率从73.2%降低到8.7%，同时保持94.3%的基线任务性能。我们发布了我们的基准数据集和防御实施，以支持未来的人工智能代理安全研究。



## **8. Taxonomy, Evaluation and Exploitation of IPI-Centric LLM Agent Defense Frameworks**

以IPI为中心的LLM代理防御框架的分类、评估和开发 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15203v1) [paper-pdf](https://arxiv.org/pdf/2511.15203v1)

**Authors**: Zimo Ji, Xunguang Wang, Zongjie Li, Pingchuan Ma, Yudong Gao, Daoyuan Wu, Xincheng Yan, Tian Tian, Shuai Wang

**Abstract**: Large Language Model (LLM)-based agents with function-calling capabilities are increasingly deployed, but remain vulnerable to Indirect Prompt Injection (IPI) attacks that hijack their tool calls. In response, numerous IPI-centric defense frameworks have emerged. However, these defenses are fragmented, lacking a unified taxonomy and comprehensive evaluation. In this Systematization of Knowledge (SoK), we present the first comprehensive analysis of IPI-centric defense frameworks. We introduce a comprehensive taxonomy of these defenses, classifying them along five dimensions. We then thoroughly assess the security and usability of representative defense frameworks. Through analysis of defensive failures in the assessment, we identify six root causes of defense circumvention. Based on these findings, we design three novel adaptive attacks that significantly improve attack success rates targeting specific frameworks, demonstrating the severity of the flaws in these defenses. Our paper provides a foundation and critical insights for the future development of more secure and usable IPI-centric agent defense frameworks.

摘要: 具有函数调用功能的基于大型语言模型（LLM）的代理被越来越多地部署，但仍然容易受到劫持其工具调用的间接提示注入（IPI）攻击。作为回应，出现了许多以IPI为中心的防御框架。然而，这些防御措施支离破碎，缺乏统一的分类和全面的评估。在本知识系统化（SoK）中，我们首次对以IPI为中心的防御框架进行了全面分析。我们对这些防御系统进行了全面的分类，并将它们按五个维度进行了分类。然后，我们彻底评估代表性防御框架的安全性和可用性。通过对评估中防御失败的分析，我们确定了规避防御的六个根本原因。基于这些发现，我们设计了三种新颖的自适应攻击，它们显着提高了针对特定框架的攻击成功率，并证明了这些防御中缺陷的严重性。我们的论文为未来开发更安全和可用的以IP为中心的代理防御框架提供了基础和重要见解。



## **9. As If We've Met Before: LLMs Exhibit Certainty in Recognizing Seen Files**

就像我们以前见过一样：法学硕士在识别可见文件方面表现出脆弱性 cs.AI

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15192v2) [paper-pdf](https://arxiv.org/pdf/2511.15192v2)

**Authors**: Haodong Li, Jingqi Zhang, Xiao Cheng, Peihua Mai, Haoyu Wang, Yan Pang

**Abstract**: The remarkable language ability of Large Language Models (LLMs) stems from extensive training on vast datasets, often including copyrighted material, which raises serious concerns about unauthorized use. While Membership Inference Attacks (MIAs) offer potential solutions for detecting such violations, existing approaches face critical limitations and challenges due to LLMs' inherent overconfidence, limited access to ground truth training data, and reliance on empirically determined thresholds.   We present COPYCHECK, a novel framework that leverages uncertainty signals to detect whether copyrighted content was used in LLM training sets. Our method turns LLM overconfidence from a limitation into an asset by capturing uncertainty patterns that reliably distinguish between ``seen" (training data) and ``unseen" (non-training data) content. COPYCHECK further implements a two-fold strategy: (1) strategic segmentation of files into smaller snippets to reduce dependence on large-scale training data, and (2) uncertainty-guided unsupervised clustering to eliminate the need for empirically tuned thresholds. Experiment results show that COPYCHECK achieves an average balanced accuracy of 90.1% on LLaMA 7b and 91.6% on LLaMA2 7b in detecting seen files. Compared to the SOTA baseline, COPYCHECK achieves over 90% relative improvement, reaching up to 93.8\% balanced accuracy. It further exhibits strong generalizability across architectures, maintaining high performance on GPT-J 6B. This work presents the first application of uncertainty for copyright detection in LLMs, offering practical tools for training data transparency.

摘要: 大型语言模型（LLM）卓越的语言能力源于对大量数据集的广泛训练，这些数据集通常包括受版权保护的材料，这引起了对未经授权使用的严重担忧。虽然成员关系推理攻击（MIA）提供了检测此类违规行为的潜在解决方案，但由于LLM固有的过度自信，对地面真实训练数据的有限访问以及对经验确定的阈值的依赖，现有方法面临着严重的限制和挑战。   我们提出了一个新的框架，利用不确定性信号来检测LLM训练集中是否使用了版权内容。我们的方法将LLM过度自信从一个限制变成一个资产，通过捕获不确定性模式，可靠地区分“看到”（训练数据）和“看不见”（非训练数据）的内容。COPYRIGHT进一步实现了双重策略：（1）将文件战略性地分割成较小的片段，以减少对大规模训练数据的依赖，以及（2）不确定性引导的无监督聚类，以消除对经验调整阈值的需求。实验结果表明，COPYRIGHT算法在LLaMA 7 b和LLaMA 2 7 b上检测可见文件的平均均衡准确率分别达到90.1%和91.6%。与SOTA基线相比，COPYRIGHT实现了90%以上的相对改进，达到93.8%的平衡精度。它还表现出跨架构的强大通用性，在GPT-J 6 B上保持高性能。这项工作首次将不确定性应用于LLM中的版权检测，为训练数据透明度提供了实用工具。



## **10. Can MLLMs Detect Phishing? A Comprehensive Security Benchmark Suite Focusing on Dynamic Threats and Multimodal Evaluation in Academic Environments**

MLLM可以检测网络钓鱼吗？专注于学术环境中的动态威胁和多模式评估的全面安全基准套件 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15165v1) [paper-pdf](https://arxiv.org/pdf/2511.15165v1)

**Authors**: Jingzhuo Zhou

**Abstract**: The rapid proliferation of Multimodal Large Language Models (MLLMs) has introduced unprecedented security challenges, particularly in phishing detection within academic environments. Academic institutions and researchers are high-value targets, facing dynamic, multilingual, and context-dependent threats that leverage research backgrounds, academic collaborations, and personal information to craft highly tailored attacks. Existing security benchmarks largely rely on datasets that do not incorporate specific academic background information, making them inadequate for capturing the evolving attack patterns and human-centric vulnerability factors specific to academia. To address this gap, we present AdapT-Bench, a unified methodological framework and benchmark suite for systematically evaluating MLLM defense capabilities against dynamic phishing attacks in academic settings.

摘要: 多模式大型语言模型（MLLM）的迅速普及带来了前所未有的安全挑战，特别是在学术环境中的网络钓鱼检测方面。学术机构和研究人员是高价值目标，面临着动态、多语言和取决于上下文的威胁，这些威胁利用研究背景、学术合作和个人信息来策划高度定制的攻击。现有的安全基准在很大程度上依赖于不包含特定学术背景信息的数据集，这使得它们不足以捕捉不断变化的攻击模式和学术界特有的以人为本的脆弱性因素。为了解决这一差距，我们提出了AdapT-Bench，这是一个统一的方法框架和基准套件，用于系统性评估MLLM防御能力在学术环境中抵御动态网络钓鱼攻击。



## **11. Unified Defense for Large Language Models against Jailbreak and Fine-Tuning Attacks in Education**

统一防御大型语言模型，防止教育领域的越狱和微调攻击 cs.CL

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14423v1) [paper-pdf](https://arxiv.org/pdf/2511.14423v1)

**Authors**: Xin Yi, Yue Li, Dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Large Language Models (LLMs) are increasingly integrated into educational applications. However, they remain vulnerable to jailbreak and fine-tuning attacks, which can compromise safety alignment and lead to harmful outputs. Existing studies mainly focus on general safety evaluations, with limited attention to the unique safety requirements of educational scenarios. To address this gap, we construct EduHarm, a benchmark containing safe-unsafe instruction pairs across five representative educational scenarios, enabling systematic safety evaluation of educational LLMs. Furthermore, we propose a three-stage shield framework (TSSF) for educational LLMs that simultaneously mitigates both jailbreak and fine-tuning attacks. First, safety-aware attention realignment redirects attention toward critical unsafe tokens, thereby restoring the harmfulness feature that discriminates between unsafe and safe inputs. Second, layer-wise safety judgment identifies harmfulness features by aggregating safety cues across multiple layers to detect unsafe instructions. Finally, defense-driven dual routing separates safe and unsafe queries, ensuring normal processing for benign inputs and guarded responses for harmful ones. Extensive experiments across eight jailbreak attack strategies demonstrate that TSSF effectively strengthens safety while preventing over-refusal of benign queries. Evaluations on three fine-tuning attack datasets further show that it consistently achieves robust defense against harmful queries while maintaining preserving utility gains from benign fine-tuning.

摘要: 大型语言模型（LLM）越来越多地集成到教育应用程序中。然而，它们仍然容易受到越狱和微调攻击，这可能会损害安全一致并导致有害输出。现有的研究主要关注一般安全评估，对教育场景独特的安全要求的关注有限。为了解决这一差距，我们构建了EduHarm，这是一个包含五种代表性教育场景中安全与不安全指令对的基准，能够对教育学LLM进行系统性安全评估。此外，我们为教育LLM提出了一个三阶段盾牌框架（TSSF），该框架同时减轻越狱和微调攻击。首先，安全意识的注意力重新调整将注意力重新引导到关键的不安全代币上，从而恢复区分不安全和安全输入的有害性特征。其次，分层安全判断通过聚集多层安全线索来检测不安全指令来识别有害特征。最后，防御驱动的双重路由将安全和不安全的查询分开，确保良性输入的正常处理和有害输入的受保护响应。针对八种越狱攻击策略的广泛实验表明，TSSF有效地增强了安全性，同时防止了对良性查询的过度拒绝。对三个微调攻击数据集的评估进一步表明，它始终实现了针对有害查询的强大防御，同时保持良性微调的效用收益。



## **12. Beyond Fixed and Dynamic Prompts: Embedded Jailbreak Templates for Advancing LLM Security**

超越固定和动态预算：用于提高LLM安全性的嵌入式越狱模板 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14140v1) [paper-pdf](https://arxiv.org/pdf/2511.14140v1)

**Authors**: Hajun Kim, Hyunsik Na, Daeseon Choi

**Abstract**: As the use of large language models (LLMs) continues to expand, ensuring their safety and robustness has become a critical challenge. In particular, jailbreak attacks that bypass built-in safety mechanisms are increasingly recognized as a tangible threat across industries, driving the need for diverse templates to support red-teaming efforts and strengthen defensive techniques. However, current approaches predominantly rely on two limited strategies: (i) substituting harmful queries into fixed templates, and (ii) having the LLM generate entire templates, which often compromises intent clarity and reproductibility. To address this gap, this paper introduces the Embedded Jailbreak Template, which preserves the structure of existing templates while naturally embedding harmful queries within their context. We further propose a progressive prompt-engineering methodology to ensure template quality and consistency, alongside standardized protocols for generation and evaluation. Together, these contributions provide a benchmark that more accurately reflects real-world usage scenarios and harmful intent, facilitating its application in red-teaming and policy regression testing.

摘要: 随着大型语言模型（LLM）的使用不断扩大，确保其安全性和稳健性已成为一项关键挑战。特别是，绕过内置安全机制的越狱攻击越来越被视为跨行业的有形威胁，这促使人们需要多样化的模板来支持红色团队工作并加强防御技术。然而，当前的方法主要依赖于两种有限的策略：（i）将有害查询替换为固定模板，以及（ii）让LLM生成整个模板，这通常会损害意图的清晰性和可重复性。为了解决这一差距，本文引入了嵌入式越狱模板，它保留了现有模板的结构，同时自然地将有害查询嵌入到其上下文中。我们进一步提出了一种渐进的预算工程方法，以确保模板质量和一致性，以及用于生成和评估的标准化协议。这些贡献共同提供了一个更准确地反映现实世界使用场景和有害意图的基准，促进其在红色团队和政策回归测试中的应用。



## **13. GRPO Privacy Is at Risk: A Membership Inference Attack Against Reinforcement Learning With Verifiable Rewards**

GRPO隐私面临风险：针对强化学习的会员推断攻击，具有可验证奖励 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14045v1) [paper-pdf](https://arxiv.org/pdf/2511.14045v1)

**Authors**: Yule Liu, Heyi Zhang, Jinyi Zheng, Zhen Sun, Zifan Peng, Tianshuo Cong, Yilong Yang, Xinlei He, Zhuo Ma

**Abstract**: Membership inference attacks (MIAs) on large language models (LLMs) pose significant privacy risks across various stages of model training. Recent advances in Reinforcement Learning with Verifiable Rewards (RLVR) have brought a profound paradigm shift in LLM training, particularly for complex reasoning tasks. However, the on-policy nature of RLVR introduces a unique privacy leakage pattern: since training relies on self-generated responses without fixed ground-truth outputs, membership inference must now determine whether a given prompt (independent of any specific response) is used during fine-tuning. This creates a threat where leakage arises not from answer memorization.   To audit this novel privacy risk, we propose Divergence-in-Behavior Attack (DIBA), the first membership inference framework specifically designed for RLVR. DIBA shifts the focus from memorization to behavioral change, leveraging measurable shifts in model behavior across two axes: advantage-side improvement (e.g., correctness gain) and logit-side divergence (e.g., policy drift). Through comprehensive evaluations, we demonstrate that DIBA significantly outperforms existing baselines, achieving around 0.8 AUC and an order-of-magnitude higher TPR@0.1%FPR. We validate DIBA's superiority across multiple settings--including in-distribution, cross-dataset, cross-algorithm, black-box scenarios, and extensions to vision-language models. Furthermore, our attack remains robust under moderate defensive measures.   To the best of our knowledge, this is the first work to systematically analyze privacy vulnerabilities in RLVR, revealing that even in the absence of explicit supervision, training data exposure can be reliably inferred through behavioral traces.

摘要: 对大型语言模型（LLM）的成员推断攻击（MIA）在模型训练的各个阶段都会带来重大的隐私风险。带可验证奖励的强化学习（WLVR）的最新进展给LLM培训带来了深刻的范式转变，特别是对于复杂的推理任务。然而，WLVR的政策性引入了一种独特的隐私泄露模式：由于训练依赖于自我生成的响应，而没有固定的地面真相输出，因此成员资格推断现在必须确定在微调期间是否使用给定的提示（独立于任何特定的响应）。这造成了一种威胁，其中泄漏不是由答案记忆引起的。   为了审计这种新颖的隐私风险，我们提出了行为分歧攻击（DIBA），这是第一个专门为WLVR设计的成员资格推断框架。DIBA将重点从记忆转移到行为改变，利用模型行为在两个轴上的可测量变化：员工端改进（例如，正确性收益）和逻辑端分歧（例如，政策漂移）。通过全面评估，我们证明DIBA的表现显着优于现有基线，实现了约0.8 AUT和更高数量级的TPR@0.1%FPR。我们验证了DIBA在多种环境中的优势--包括内分布、跨数据集、跨算法、黑匣子场景以及视觉语言模型的扩展。此外，在适度的防御措施下，我们的攻击仍然强劲。   据我们所知，这是第一个系统性分析WLVR中隐私漏洞的工作，揭示了即使在缺乏明确监督的情况下，也可以通过行为痕迹可靠地推断训练数据暴露。



## **14. Jailbreaking Large Vision Language Models in Intelligent Transportation Systems**

突破智能交通系统中的大视觉语言模型 cs.AI

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13892v1) [paper-pdf](https://arxiv.org/pdf/2511.13892v1)

**Authors**: Badhan Chandra Das, Md Tasnim Jawad, Md Jueal Mia, M. Hadi Amini, Yanzhao Wu

**Abstract**: Large Vision Language Models (LVLMs) demonstrate strong capabilities in multimodal reasoning and many real-world applications, such as visual question answering. However, LVLMs are highly vulnerable to jailbreaking attacks. This paper systematically analyzes the vulnerabilities of LVLMs integrated in Intelligent Transportation Systems (ITS) under carefully crafted jailbreaking attacks. First, we carefully construct a dataset with harmful queries relevant to transportation, following OpenAI's prohibited categories to which the LVLMs should not respond. Second, we introduce a novel jailbreaking attack that exploits the vulnerabilities of LVLMs through image typography manipulation and multi-turn prompting. Third, we propose a multi-layered response filtering defense technique to prevent the model from generating inappropriate responses. We perform extensive experiments with the proposed attack and defense on the state-of-the-art LVLMs (both open-source and closed-source). To evaluate the attack method and defense technique, we use GPT-4's judgment to determine the toxicity score of the generated responses, as well as manual verification. Further, we compare our proposed jailbreaking method with existing jailbreaking techniques and highlight severe security risks involved with jailbreaking attacks with image typography manipulation and multi-turn prompting in the LVLMs integrated in ITS.

摘要: 大型视觉语言模型（LVLM）在多模式推理和许多现实世界应用（例如视觉问答）方面表现出强大的能力。然而，LVLM极易受到越狱攻击。本文系统地分析了智能交通系统（ITS）中集成的LVLM在精心设计的越狱攻击下的漏洞。首先，我们仔细构建一个包含与交通相关的有害查询的数据集，遵循OpenAI禁止的LVLM不应响应的类别。其次，我们引入了一种新颖的越狱攻击，该攻击通过图像印刷操作和多圈提示来利用LVLM的漏洞。第三，我们提出了一种多层响应过滤防御技术，以防止模型产生不适当的响应。我们进行了广泛的实验，提出的攻击和防御的最先进的LVLM（开源和闭源）。为了评估攻击方法和防御技术，我们使用GPT-4的判断来确定生成的响应的毒性分数，以及人工验证。此外，我们将我们提出的越狱方法与现有的越狱技术进行比较，并强调在集成在ITS中的LVLM中使用图像排版操作和多回合提示的越狱攻击所涉及的严重安全风险。



## **15. ForgeDAN: An Evolutionary Framework for Jailbreaking Aligned Large Language Models**

ForgeDAN：越狱对齐大型语言模型的进化框架 cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13548v1) [paper-pdf](https://arxiv.org/pdf/2511.13548v1)

**Authors**: Siyang Cheng, Gaotian Liu, Rui Mei, Yilin Wang, Kejia Zhang, Kaishuo Wei, Yuqi Yu, Weiping Wen, Xiaojie Wu, Junhua Liu

**Abstract**: The rapid adoption of large language models (LLMs) has brought both transformative applications and new security risks, including jailbreak attacks that bypass alignment safeguards to elicit harmful outputs. Existing automated jailbreak generation approaches e.g. AutoDAN, suffer from limited mutation diversity, shallow fitness evaluation, and fragile keyword-based detection. To address these limitations, we propose ForgeDAN, a novel evolutionary framework for generating semantically coherent and highly effective adversarial prompts against aligned LLMs. First, ForgeDAN introduces multi-strategy textual perturbations across \textit{character, word, and sentence-level} operations to enhance attack diversity; then we employ interpretable semantic fitness evaluation based on a text similarity model to guide the evolutionary process toward semantically relevant and harmful outputs; finally, ForgeDAN integrates dual-dimensional jailbreak judgment, leveraging an LLM-based classifier to jointly assess model compliance and output harmfulness, thereby reducing false positives and improving detection effectiveness. Our evaluation demonstrates ForgeDAN achieves high jailbreaking success rates while maintaining naturalness and stealth, outperforming existing SOTA solutions.

摘要: 大型语言模型（LLM）的迅速采用既带来了变革性的应用程序，也带来了新的安全风险，包括绕过对齐保障措施以引发有害输出的越狱攻击。现有的自动越狱生成方法（例如AutoDAN）存在突变多样性有限、适应度评估浅和基于关键字的脆弱检测的问题。为了解决这些限制，我们提出了ForgeDAN，这是一种新颖的进化框架，用于针对对齐的LLM生成语义一致且高效的对抗性提示。首先，ForgeDAN在\textit{字符、单词和会话级别}操作中引入多策略文本扰动，以增强攻击多样性;然后我们基于文本相似性模型采用可解释的语义适应度评估来引导进化过程走向语义相关和有害的输出;最后，ForgeDAN集成了二维越狱判断，利用基于LLM的分类器来联合评估模型合规性和输出危害性，从而减少假阳性并提高检测有效性。我们的评估表明，ForgeDAN在保持自然性和隐形性的同时实现了很高的越狱成功率，优于现有的SOTA解决方案。



## **16. Tight and Practical Privacy Auditing for Differentially Private In-Context Learning**

针对差异私密的上下文学习进行严格而实用的隐私审计 cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13502v1) [paper-pdf](https://arxiv.org/pdf/2511.13502v1)

**Authors**: Yuyang Xia, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) perform in-context learning (ICL) by adapting to tasks from prompt demonstrations, which in practice often contain private or proprietary data. Although differential privacy (DP) with private voting is a pragmatic mitigation, DP-ICL implementations are error-prone, and worst-case DP bounds may substantially overestimate actual leakage, calling for practical auditing tools. We present a tight and efficient privacy auditing framework for DP-ICL systems that runs membership inference attacks and translates their success rates into empirical privacy guarantees using Gaussian DP. Our analysis of the private voting mechanism identifies vote configurations that maximize the auditing signal, guiding the design of audit queries that reliably reveal whether a canary demonstration is present in the context. The framework supports both black-box (API-only) and white-box (internal vote) threat models, and unifies auditing for classification and generation by reducing both to a binary decision problem. Experiments on standard text classification and generation benchmarks show that our empirical leakage estimates closely match theoretical DP budgets on classification tasks and are consistently lower on generation tasks due to conservative embedding-sensitivity bounds, making our framework a practical privacy auditor and verifier for real-world DP-ICL deployments.

摘要: 大型语言模型（LLM）通过适应即时演示的任务来执行上下文学习（ICL），这些任务在实践中通常包含私人或专有数据。尽管带有私人投票的差异隐私（DP）是一种务实的缓解措施，但DP-ICL的实现很容易出错，而且最坏情况下的DP界限可能会大大高估实际泄漏，因此需要实用的审计工具。我们为DP-ICL系统提供了一个严格而高效的隐私审计框架，该框架运行成员资格推断攻击，并使用高斯DP将其成功率转化为经验隐私保证。我们对私人投票机制的分析确定了最大化审计信号的投票配置，指导审计查询的设计，可靠地揭示上下文中是否存在金丝雀演示。该框架支持黑匣子（仅API）和白盒（内部投票）威胁模型，并通过将两者简化为二元决策问题来统一分类和生成审计。标准文本分类和生成基准的实验表明，我们的经验泄露估计与分类任务的理论DP预算密切匹配，并且由于保守的嵌入敏感性界限，生成任务的泄漏估计始终较低，使我们的框架成为现实世界DP-ICL部署的实用隐私审计器和验证器。



## **17. An LLM-based Quantitative Framework for Evaluating High-Stealthy Backdoor Risks in OSS Supply Chains**

评估OSS供应链中高隐形后门风险的基于法学硕士的量化框架 cs.SE

7 figures, 4 tables, conference

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13341v1) [paper-pdf](https://arxiv.org/pdf/2511.13341v1)

**Authors**: Zihe Yan, Kai Luo, Haoyu Yang, Yang Yu, Zhuosheng Zhang, Guancheng Li

**Abstract**: In modern software development workflows, the open-source software supply chain contributes significantly to efficient and convenient engineering practices. With increasing system complexity, using open-source software as third-party dependencies has become a common practice. However, the lack of maintenance for underlying dependencies and insufficient community auditing create challenges in ensuring source code security and the legitimacy of repository maintainers, especially under high-stealthy backdoor attacks exemplified by the XZ-Util incident. To address these problems, we propose a fine-grained project evaluation framework for backdoor risk assessment in open-source software. The framework models stealthy backdoor attacks from the viewpoint of the attacker and defines targeted metrics for each attack stage. In addition, to overcome the limitations of static analysis in assessing the reliability of repository maintenance activities such as irregular committer privilege escalation and limited participation in reviews, the framework uses large language models (LLMs) to conduct semantic evaluation of code repositories without relying on manually crafted patterns. The framework is evaluated on sixty six high-priority packages in the Debian ecosystem. The experimental results indicate that the current open-source software supply chain is exposed to various security risks.

摘要: 在现代软件开发工作流程中，开源软件供应链对高效和便捷的工程实践做出了重大贡献。随着系统复杂性的增加，使用开源软件作为第三方依赖项已成为一种普遍做法。然而，底层依赖项缺乏维护和社区审计不足给确保源代码安全和存储库维护者的合法性带来了挑战，特别是在以XZ-Usil事件为例的高度隐蔽的后门攻击下。为了解决这些问题，我们提出了一个细粒度的项目评估框架，用于开源软件中的后门风险评估。该框架从攻击者的角度对隐形后门攻击进行建模，并为每个攻击阶段定义有针对性的指标。此外，为了克服静态分析在评估存储库维护活动的可靠性方面的局限性，例如不规则的提交者特权升级和有限的参与审查，该框架使用大型语言模型（LLM）来对代码存储库进行语义评估，而不依赖于手工制作的模式。该框架在Debian生态系统中的66个高优先级包上进行了评估。实验结果表明，当前开源软件供应链面临着各种安全风险。



## **18. Shedding Light on VLN Robustness: A Black-box Framework for Indoor Lighting-based Adversarial Attack**

VLN鲁棒性的减弱：基于室内照明的对抗攻击的黑匣子框架 cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.13132v1) [paper-pdf](https://arxiv.org/pdf/2511.13132v1)

**Authors**: Chenyang Li, Wenbing Tang, Yihao Huang, Sinong Simon Zhan, Ming Hu, Xiaojun Jia, Yang Liu

**Abstract**: Vision-and-Language Navigation (VLN) agents have made remarkable progress, but their robustness remains insufficiently studied. Existing adversarial evaluations often rely on perturbations that manifest as unusual textures rarely encountered in everyday indoor environments. Errors under such contrived conditions have limited practical relevance, as real-world agents are unlikely to encounter such artificial patterns. In this work, we focus on indoor lighting, an intrinsic yet largely overlooked scene attribute that strongly influences navigation. We propose Indoor Lighting-based Adversarial Attack (ILA), a black-box framework that manipulates global illumination to disrupt VLN agents. Motivated by typical household lighting usage, we design two attack modes: Static Indoor Lighting-based Attack (SILA), where the lighting intensity remains constant throughout an episode, and Dynamic Indoor Lighting-based Attack (DILA), where lights are switched on or off at critical moments to induce abrupt illumination changes. We evaluate ILA on two state-of-the-art VLN models across three navigation tasks. Results show that ILA significantly increases failure rates while reducing trajectory efficiency, revealing previously unrecognized vulnerabilities of VLN agents to realistic indoor lighting variations.

摘要: 视觉与语言导航（VLN）代理已经取得了显着的进步，但其稳健性仍然研究不足。现有的对抗性评估通常依赖于扰动，这些扰动表现为日常室内环境中很少遇到的异常纹理。这种人为条件下的错误的实际意义有限，因为现实世界的代理人不太可能遇到这种人为模式。在这项工作中，我们重点关注室内照明，这是一种固有但在很大程度上被忽视的场景属性，它强烈影响导航。我们提出了基于室内照明的对抗攻击（ILA），这是一种黑匣子框架，可以操纵全球照明来扰乱VLN代理。受典型家庭照明使用的启发，我们设计了两种攻击模式：静态室内照明攻击（SILA），其中照明强度在整个剧集中保持恒定，以及动态室内照明攻击（DILA），其中在关键时刻打开或关闭灯光以引发突然的照明变化。我们在三个导航任务中评估了两个最先进的VLN模型的ILA。结果表明，ILA显着增加了故障率，同时降低了轨迹效率，揭示了VLN代理对现实室内照明变化的脆弱性。



## **19. LLM Reinforcement in Context**

LLM在上下文中的强化 cs.CL

4 pages

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12782v1) [paper-pdf](https://arxiv.org/pdf/2511.12782v1)

**Authors**: Thomas Rivasseau

**Abstract**: Current Large Language Model alignment research mostly focuses on improving model robustness against adversarial attacks and misbehavior by training on examples and prompting. Research has shown that LLM jailbreak probability increases with the size of the user input or conversation length. There is a lack of appropriate research into means of strengthening alignment which also scale with user input length. We propose interruptions as a possible solution to this problem. Interruptions are control sentences added to the user input approximately every x tokens for some arbitrary x. We suggest that this can be generalized to the Chain-of-Thought process to prevent scheming.

摘要: 当前的大型语言模型对齐研究主要集中在通过对示例和提示进行训练来提高模型对对抗性攻击和不当行为的稳健性。研究表明，LLM越狱概率随着用户输入或对话长度的大小而增加。缺乏对加强对齐的方法进行适当的研究，而对齐也随用户输入长度而变化。我们建议中断作为这个问题的一种可能的解决方案。中断是针对某个任意x，大约每x个记号添加到用户输入中的控制句。我们建议将其推广到思想链过程中，以防止阴谋。



## **20. Whose Narrative is it Anyway? A KV Cache Manipulation Attack**

这到底是谁的叙述？KV缓存操纵攻击 cs.CR

7 pages, 10 figures

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12752v1) [paper-pdf](https://arxiv.org/pdf/2511.12752v1)

**Authors**: Mukkesh Ganesh, Kaushik Iyer, Arun Baalaaji Sankar Ananthan

**Abstract**: The Key Value(KV) cache is an important component for efficient inference in autoregressive Large Language Models (LLMs), but its role as a representation of the model's internal state makes it a potential target for integrity attacks. This paper introduces "History Swapping," a novel block-level attack that manipulates the KV cache to steer model generation without altering the user-facing prompt. The attack involves overwriting a contiguous segment of the active generation's cache with a precomputed cache from a different topic. We empirically evaluate this method across 324 configurations on the Qwen 3 family of models, analyzing the impact of timing, magnitude, and layer depth of the cache overwrite. Our findings reveal that only full-layer overwrites can successfully hijack the conversation's topic, leading to three distinct behaviors: immediate and persistent topic shift, partial recovery, or a delayed hijack. Furthermore, we observe that high-level structural plans are encoded early in the generation process and local discourse structure is maintained by the final layers of the model. This work demonstrates that the KV cache is a significant vector for security analysis, as it encodes not just context but also topic trajectory and structural planning, making it a powerful interface for manipulating model behavior.

摘要: Key Value（KV）缓存是自回归大型语言模型（LLM）中高效推理的重要组件，但它作为模型内部状态的表示的角色使其成为完整性攻击的潜在目标。本文介绍了“历史交换”，这是一种新型的块级攻击，它操纵KV缓存来引导模型生成，而不改变面向用户的提示。该攻击涉及使用来自不同主题的预先计算的缓存来同步活动代缓存的连续段。我们在Qwen 3系列模型的324种配置上对该方法进行了经验评估，分析了缓存重写的时间、幅度和层深度的影响。我们的研究结果表明，只有全层覆盖可以成功劫持会话的主题，导致三种不同的行为：立即和持久的主题转移，部分恢复，或延迟劫持。此外，我们观察到，高层次的结构计划编码早期的生成过程和本地话语结构是由模型的最后几层。这项工作表明KV缓存是安全分析的重要载体，因为它不仅编码上下文，还编码主题轨迹和结构规划，使其成为操纵模型行为的强大接口。



## **21. Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs**

进化方法，而不是预言：对LLM越狱攻击的进化综合 cs.CL

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12710v1) [paper-pdf](https://arxiv.org/pdf/2511.12710v1)

**Authors**: Yunhao Chen, Xin Wang, Juncheng Li, Yixu Wang, Jie Li, Yan Teng, Yingchun Wang, Xingjun Ma

**Abstract**: Automated red teaming frameworks for Large Language Models (LLMs) have become increasingly sophisticated, yet they share a fundamental limitation: their jailbreak logic is confined to selecting, combining, or refining pre-existing attack strategies. This binds their creativity and leaves them unable to autonomously invent entirely new attack mechanisms. To overcome this gap, we introduce \textbf{EvoSynth}, an autonomous framework that shifts the paradigm from attack planning to the evolutionary synthesis of jailbreak methods. Instead of refining prompts, EvoSynth employs a multi-agent system to autonomously engineer, evolve, and execute novel, code-based attack algorithms. Crucially, it features a code-level self-correction loop, allowing it to iteratively rewrite its own attack logic in response to failure. Through extensive experiments, we demonstrate that EvoSynth not only establishes a new state-of-the-art by achieving an 85.5\% Attack Success Rate (ASR) against highly robust models like Claude-Sonnet-4.5, but also generates attacks that are significantly more diverse than those from existing methods. We release our framework to facilitate future research in this new direction of evolutionary synthesis of jailbreak methods. Code is available at: https://github.com/dongdongunique/EvoSynth.

摘要: 大型语言模型（LLM）的自动化红色团队框架已变得越来越复杂，但它们都有一个根本性的局限性：它们的越狱逻辑仅限于选择、组合或完善预先存在的攻击策略。这束缚了他们的创造力，使他们无法自主发明全新的攻击机制。为了克服这一差距，我们引入了\textBF{EvoSynth}，这是一个自主框架，将范式从攻击规划转变为越狱方法的进化合成。EvoSynth没有细化提示，而是采用多代理系统来自主设计、进化和执行新颖的基于代码的攻击算法。至关重要的是，它具有代码级自校正循环，允许它迭代重写自己的攻击逻辑以响应失败。通过大量的实验，我们证明了EvoSynth不仅通过实现85.5%的攻击成功率（ASR）建立了一个新的最先进的技术，对像Claude-Sonnet-4.5这样的高度鲁棒的模型，而且还生成了比现有方法更多样化的攻击。我们发布我们的框架，以促进未来的研究在这个新的方向进化合成越狱方法。代码可访问：https://github.com/dongdongunique/EvoSynth。



## **22. Uncovering and Aligning Anomalous Attention Heads to Defend Against NLP Backdoor Attacks**

发现并协调异常注意力以防御NLP后门攻击 cs.CR

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.13789v1) [paper-pdf](https://arxiv.org/pdf/2511.13789v1)

**Authors**: Haotian Jin, Yang Li, Haihui Fan, Lin Shen, Xiangfang Li, Bo Li

**Abstract**: Backdoor attacks pose a serious threat to the security of large language models (LLMs), causing them to exhibit anomalous behavior under specific trigger conditions. The design of backdoor triggers has evolved from fixed triggers to dynamic or implicit triggers. This increased flexibility in trigger design makes it challenging for defenders to identify their specific forms accurately. Most existing backdoor defense methods are limited to specific types of triggers or rely on an additional clean model for support. To address this issue, we propose a backdoor detection method based on attention similarity, enabling backdoor detection without prior knowledge of the trigger. Our study reveals that models subjected to backdoor attacks exhibit unusually high similarity among attention heads when exposed to triggers. Based on this observation, we propose an attention safety alignment approach combined with head-wise fine-tuning to rectify potentially contaminated attention heads, thereby effectively mitigating the impact of backdoor attacks. Extensive experimental results demonstrate that our method significantly reduces the success rate of backdoor attacks while preserving the model's performance on downstream tasks.

摘要: 后门攻击对大型语言模型（LLM）的安全性构成严重威胁，导致它们在特定触发条件下表现出异常行为。后门触发器的设计已经从固定触发器发展到动态或隐式触发器。触发器设计的灵活性增加，使防御者难以准确识别其特定形式。大多数现有的后门防御方法仅限于特定类型的触发器或依赖于额外的干净模型来支持。为了解决这个问题，我们提出了一种基于注意力相似性的后门检测方法，在不了解触发器的情况下实现后门检测。我们的研究表明，遭受后门攻击的模型在暴露于触发器时，注意力头之间表现出异常高的相似性。基于这一观察，我们提出了一种注意力安全调整方法，结合头部微调，以纠正可能被污染的注意力头部，从而有效减轻后门攻击的影响。大量的实验结果表明，我们的方法显着降低了后门攻击的成功率，同时保留了模型在下游任务上的性能。



## **23. Scaling Patterns in Adversarial Alignment: Evidence from Multi-LLM Jailbreak Experiments**

对抗性对齐中的缩放模式：来自Multi-LLM越狱实验的证据 cs.LG

19 pages, 6 figures, 3 tables

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.13788v1) [paper-pdf](https://arxiv.org/pdf/2511.13788v1)

**Authors**: Samuel Nathanson, Rebecca Williams, Cynthia Matuszek

**Abstract**: Large language models (LLMs) increasingly operate in multi-agent and safety-critical settings, raising open questions about how their vulnerabilities scale when models interact adversarially. This study examines whether larger models can systematically jailbreak smaller ones - eliciting harmful or restricted behavior despite alignment safeguards. Using standardized adversarial tasks from JailbreakBench, we simulate over 6,000 multi-turn attacker-target exchanges across major LLM families and scales (0.6B-120B parameters), measuring both harm score and refusal behavior as indicators of adversarial potency and alignment integrity. Each interaction is evaluated through aggregated harm and refusal scores assigned by three independent LLM judges, providing a consistent, model-based measure of adversarial outcomes. Aggregating results across prompts, we find a strong and statistically significant correlation between mean harm and the logarithm of the attacker-to-target size ratio (Pearson r = 0.51, p < 0.001; Spearman rho = 0.52, p < 0.001), indicating that relative model size correlates with the likelihood and severity of harmful completions. Mean harm score variance is higher across attackers (0.18) than across targets (0.10), suggesting that attacker-side behavioral diversity contributes more to adversarial outcomes than target susceptibility. Attacker refusal frequency is strongly and negatively correlated with harm (rho = -0.93, p < 0.001), showing that attacker-side alignment mitigates harmful responses. These findings reveal that size asymmetry influences robustness and provide exploratory evidence for adversarial scaling patterns, motivating more controlled investigations into inter-model alignment and safety.

摘要: 大型语言模型（LLM）越来越多地在多代理和安全关键环境中运行，这引发了有关模型对抗性交互时漏洞如何扩展的悬而未决的问题。这项研究考察了较大的模型是否可以系统性地越狱较小的模型--尽管有一致保障，但仍会引发有害或限制的行为。使用JailbreakBench的标准化对抗任务，我们模拟了主要LLM家族和量表（0.6B-120 B参数）中的6，000多个多回合攻击者-目标交换，测量伤害评分和拒绝行为作为对抗效力和对齐完整性的指标。每次互动都是通过三位独立LLM法官分配的总伤害和拒绝分数来评估的，从而提供一致的、基于模型的对抗结果测量。汇总各个提示的结果，我们发现平均伤害与攻击者与目标规模比的对数之间存在很强且具有统计学意义的相关性（Pearson r = 0.51，p < 0.001; Spearman rho = 0.52，p < 0.001），表明相对模型大小与有害完成的可能性和严重性相关。攻击者之间的平均伤害评分方差（0.18）高于目标之间的平均伤害评分方差（0.10），这表明攻击者方的行为多样性对对抗结果的贡献大于目标易感性。攻击者拒绝频率与伤害呈强烈负相关（rho =-0.93，p < 0.001），表明攻击者方的一致可以减轻有害反应。这些发现表明，尺寸不对称会影响稳健性，并为对抗性缩放模式提供探索性证据，从而激励对模型间对齐和安全性进行更受控的调查。



## **24. Beyond Pixels: Semantic-aware Typographic Attack for Geo-Privacy Protection**

超越像素：用于地理隐私保护的语义感知印刷攻击 cs.CV

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12575v1) [paper-pdf](https://arxiv.org/pdf/2511.12575v1)

**Authors**: Jiayi Zhu, Yihao Huang, Yue Cao, Xiaojun Jia, Qing Guo, Felix Juefei-Xu, Geguang Pu, Bin Wang

**Abstract**: Large Visual Language Models (LVLMs) now pose a serious yet overlooked privacy threat, as they can infer a social media user's geolocation directly from shared images, leading to unintended privacy leakage. While adversarial image perturbations provide a potential direction for geo-privacy protection, they require relatively strong distortions to be effective against LVLMs, which noticeably degrade visual quality and diminish an image's value for sharing. To overcome this limitation, we identify typographical attacks as a promising direction for protecting geo-privacy by adding text extension outside the visual content. We further investigate which textual semantics are effective in disrupting geolocation inference and design a two-stage, semantics-aware typographical attack that generates deceptive text to protect user privacy. Extensive experiments across three datasets demonstrate that our approach significantly reduces geolocation prediction accuracy of five state-of-the-art commercial LVLMs, establishing a practical and visually-preserving protection strategy against emerging geo-privacy threats.

摘要: 大型视觉语言模型（LVLM）现在构成了一个严重但被忽视的隐私威胁，因为它们可以直接从共享图像中推断社交媒体用户的地理位置，从而导致意外的隐私泄露。虽然对抗性图像扰动为地理隐私保护提供了一个潜在的方向，但它们需要相对强的失真才能有效对抗LVLM，而LVLM会显着降低视觉质量并降低图像的共享价值。为了克服这一限制，我们将印刷攻击确定为通过在视觉内容之外添加文本扩展来保护地理隐私的一个有希望的方向。我们进一步研究哪些文本语义可以有效扰乱地理位置推断，并设计一种两阶段、语义感知的印刷攻击，该攻击可以生成欺骗性文本以保护用户隐私。跨三个数据集的广泛实验表明，我们的方法显着降低了五种最先进的商业LVLM的地理位置预测准确性，建立了针对新出现的地理隐私威胁的实用且视觉保护策略。



## **25. SGuard-v1: Safety Guardrail for Large Language Models**

SGuard-v1：大型语言模型的安全保障 cs.CL

Technical Report

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12497v1) [paper-pdf](https://arxiv.org/pdf/2511.12497v1)

**Authors**: JoonHo Lee, HyeonMin Cho, Jaewoong Yun, Hyunjae Lee, JunKyu Lee, Juree Seok

**Abstract**: We present SGuard-v1, a lightweight safety guardrail for Large Language Models (LLMs), which comprises two specialized models to detect harmful content and screen adversarial prompts in human-AI conversational settings. The first component, ContentFilter, is trained to identify safety risks in LLM prompts and responses in accordance with the MLCommons hazard taxonomy, a comprehensive framework for trust and safety assessment of AI. The second component, JailbreakFilter, is trained with a carefully designed curriculum over integrated datasets and findings from prior work on adversarial prompting, covering 60 major attack types while mitigating false-unsafe classification. SGuard-v1 is built on the 2B-parameter Granite-3.3-2B-Instruct model that supports 12 languages. We curate approximately 1.4 million training instances from both collected and synthesized data and perform instruction tuning on the base model, distributing the curated data across the two component according to their designated functions. Through extensive evaluation on public and proprietary safety benchmarks, SGuard-v1 achieves state-of-the-art safety performance while remaining lightweight, thereby reducing deployment overhead. SGuard-v1 also improves interpretability for downstream use by providing multi-class safety predictions and their binary confidence scores. We release the SGuard-v1 under the Apache-2.0 License to enable further research and practical deployment in AI safety.

摘要: 我们介绍了SGuard-v1，这是一种适用于大型语言模型（LLM）的轻量级安全护栏，它包括两个专门的模型，用于检测有害内容并在人工智能对话设置中屏幕对抗性提示。第一个组件ContentLayer经过培训，能够根据MLCommons危险分类法识别LLM提示和响应中的安全风险，MLCommons危险分类法是人工智能信任和安全评估的综合框架。第二个组件JailbreakLayer是经过精心设计的课程培训的，该课程涵盖了集成的数据集和之前对抗提示工作的结果，涵盖60种主要攻击类型，同时减轻了错误不安全的分类。SGuard-v1构建在支持12种语言的2B参数Granite-3.3- 2B-Direct模型之上。我们从收集和合成的数据中策划了大约140万个训练实例，并对基本模型执行指令调优，根据其指定功能将策划的数据分布在两个组件之间。通过对公共和专有安全基准的广泛评估，SGuard-v1实现了最先进的安全性能，同时保持重量轻，从而减少了部署费用。SGuard-v1还通过提供多类别安全预测及其二进制置信度分数来提高下游使用的可解释性。我们根据Apache-2.0许可发布了SGuard-v1，以支持人工智能安全方面的进一步研究和实际部署。



## **26. GRAPHTEXTACK: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs**

GRAPHTEXTACK：对LLM增强型GNN的现实黑匣子节点注入攻击 cs.CR

AAAI 2026

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2511.12423v1) [paper-pdf](https://arxiv.org/pdf/2511.12423v1)

**Authors**: Jiaji Ma, Puja Trivedi, Danai Koutra

**Abstract**: Text-attributed graphs (TAGs), which combine structural and textual node information, are ubiquitous across many domains. Recent work integrates Large Language Models (LLMs) with Graph Neural Networks (GNNs) to jointly model semantics and structure, resulting in more general and expressive models that achieve state-of-the-art performance on TAG benchmarks. However, this integration introduces dual vulnerabilities: GNNs are sensitive to structural perturbations, while LLM-derived features are vulnerable to prompt injection and adversarial phrasing. While existing adversarial attacks largely perturb structure or text independently, we find that uni-modal attacks cause only modest degradation in LLM-enhanced GNNs. Moreover, many existing attacks assume unrealistic capabilities, such as white-box access or direct modification of graph data. To address these gaps, we propose GRAPHTEXTACK, the first black-box, multi-modal{, poisoning} node injection attack for LLM-enhanced GNNs. GRAPHTEXTACK injects nodes with carefully crafted structure and semantics to degrade model performance, operating under a realistic threat model without relying on model internals or surrogate models. To navigate the combinatorial, non-differentiable search space of connectivity and feature assignments, GRAPHTEXTACK introduces a novel evolutionary optimization framework with a multi-objective fitness function that balances local prediction disruption and global graph influence. Extensive experiments on five datasets and two state-of-the-art LLM-enhanced GNN models show that GRAPHTEXTACK significantly outperforms 12 strong baselines.

摘要: 文本属性图（TAG）结合了结构和文本节点信息，在许多领域中都无处不在。最近的工作将大型语言模型（LLM）与图形神经网络（GNN）集成，以联合建模语义和结构，从而产生更通用和更富有表达力的模型，在TAG基准测试上实现最先进的性能。然而，这种集成引入了双重漏洞：GNN对结构性扰动敏感，而LLM衍生的功能容易受到提示注入和对抗性措辞的影响。虽然现有的对抗性攻击在很大程度上独立地扰乱结构或文本，但我们发现单模式攻击只会导致LLM增强的GNN的适度降级。此外，许多现有的攻击都假设不切实际的能力，例如白盒访问或直接修改图形数据。为了解决这些差距，我们提出了GRAPHTEXTACK，这是针对LLM增强型GNN的第一个黑匣子、多模式{，中毒}节点注入攻击。GRAPHTEXTACK注入具有精心设计的结构和语义的节点，以降低模型性能，在现实的威胁模型下运行，而不依赖模型内部或代理模型。为了在连接性和特征分配的组合性、不可微搜索空间中导航，GRAPHTEXTACK引入了一种新颖的进化优化框架，该框架具有多目标适应度函数，该函数平衡了局部预测中断和全局图影响。对五个数据集和两个最先进的LLM增强GNN模型的广泛实验表明，GRAPHTEXTACK的表现显着优于12个强基线。



## **27. Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment**

差异化定向干预避免LLM安全一致的框架 cs.CR

AAAI-26-AIA

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2511.06852v3) [paper-pdf](https://arxiv.org/pdf/2511.06852v3)

**Authors**: Peng Zhang, Peijie Sun

**Abstract**: Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.

摘要: 安全一致为大型语言模型（LLM）灌输了拒绝恶意请求的关键能力。之前的作品将这种拒绝机制建模为激活空间中的单一线性方向。我们认为这是一种过于简单化的做法，将两个功能上不同的神经过程混为一谈：伤害的检测和拒绝的执行。在这项工作中，我们将这个单一的表示解构为伤害检测方向和拒绝执行方向。利用这个细粒度模型，我们引入了差异双向干预（DBDI），这是一种新的白盒框架，可以精确地中和关键层的安全对齐。DBDI对拒绝执行方向应用自适应投影无效，同时通过直接转向抑制伤害检测方向。大量实验表明，DBDI优于著名的越狱方法，对Llama-2等模型的攻击成功率高达97.88%。通过提供更细粒度和机械化的框架，我们的工作为深入了解LLM安全对齐提供了新的方向。



## **28. DRIP: Defending Prompt Injection via Token-wise Representation Editing and Residual Instruction Fusion**

DRIP：通过逐令牌表示编辑和剩余指令融合来防御提示注入 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.00447v2) [paper-pdf](https://arxiv.org/pdf/2511.00447v2)

**Authors**: Ruofan Liu, Yun Lin, Zhiyong Huang, Jin Song Dong

**Abstract**: Large language models (LLMs) are increasingly integrated into IT infrastructures, where they process user data according to predefined instructions. However, conventional LLMs remain vulnerable to prompt injection, where malicious users inject directive tokens into the data to subvert model behavior. Existing defenses train LLMs to semantically separate data and instruction tokens, but still struggle to (1) balance utility and security and (2) prevent instruction-like semantics in the data from overriding the intended instructions.   We propose DRIP, which (1) precisely removes instruction semantics from tokens in the data section while preserving their data semantics, and (2) robustly preserves the effect of the intended instruction even under strong adversarial content. To "de-instructionalize" data tokens, DRIP introduces a data curation and training paradigm with a lightweight representation-editing module that edits embeddings of instruction-like tokens in the data section, enhancing security without harming utility. To ensure non-overwritability of instructions, DRIP adds a minimal residual module that reduces the ability of adversarial data to overwrite the original instruction. We evaluate DRIP on LLaMA 8B and Mistral 7B against StruQ, SecAlign, ISE, and PFT on three prompt-injection benchmarks (SEP, AlpacaFarm, and InjecAgent). DRIP improves role-separation score by 12-49\%, reduces attack success rate by over 66\% under adaptive attacks, and matches the utility of the undefended model, establishing a new state of the art for prompt-injection robustness.

摘要: 大型语言模型（LLM）越来越多地集成到IT基础设施中，它们根据预定义的指令处理用户数据。然而，传统的LLM仍然容易受到提示注入的影响，即恶意用户将指令令牌注入到数据中以颠覆模型行为。现有的防御措施训练LLM在语义上分离数据和指令令牌，但仍然难以（1）平衡实用性和安全性，以及（2）防止数据中类似描述的语义覆盖预期指令。   我们提出了DRIP，它（1）从数据部分中的令牌中精确地删除指令语义，同时保留其数据语义，（2）即使在强对抗性内容下也能稳健地保留预期指令的效果。为了“去伪化”数据令牌，DRIP引入了一种数据策展和训练范式，该范式具有轻量级的表示编辑模块，该模块可以编辑数据部分中类似描述的令牌的嵌入，从而在不损害实用性的情况下增强了安全性。为了确保指令的不可重写性，DRIP添加了一个最小剩余模块，该模块降低了对抗数据重写原始指令的能力。我们在三个预算注入基准（SDP、AlpacaFarm和InjecAgent）上针对StruQ、SecAlign、ISE和PFT评估了LLaMA 8B和Mistral 7 B上的DRIP。DRIP将角色分离分数提高了12- 49%，在适应性攻击下将攻击成功率降低了66%以上，并匹配了无防御模型的实用性，为预算注入鲁棒性建立了新的最新水平。



## **29. SoK: Honeypots & LLMs, More Than the Sum of Their Parts?**

SoK：蜜罐和LLM，超过其部分的总和？ cs.CR

Systemization of Knowledge

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2510.25939v3) [paper-pdf](https://arxiv.org/pdf/2510.25939v3)

**Authors**: Robert A. Bridges, Thomas R. Mitchell, Mauricio Muñoz, Ted Henriksson

**Abstract**: The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design, achieving high-fidelity deception with low operational risk. Through a flurry of research since late 2022, steady progress from ideation to prototype implementation is exhibited. Since late 2022, a flurry of research has demonstrated steady progress from ideation to prototype implementation. While promising, evaluations show only incremental progress in real-world deployments, and the field still lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview and analysis of this new domain. We survey and systematize the field by focusing on three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-powered honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.

摘要: 大型语言模型（LLM）的出现有望解决蜜罐设计中长期存在的悖论，以低操作风险实现高保真欺骗。通过自2022年底以来的一系列研究，从构思到原型实现的稳步进展已经显现。自2022年底以来，一系列研究表明，从构思到原型实施正在稳步进展。虽然有希望，但评估仅显示现实世界部署中的渐进进展，并且该领域仍然缺乏对新兴架构模式、核心挑战和评估范式的一致理解。为了填补这一空白，这篇知识系统化（SoK）论文首次对这一新领域进行了全面的概述和分析。我们通过关注三个关键的交叉研究领域来调查和系统化该领域：首先，我们提供了蜜罐检测向量的分类，构建了基于LLM的现实主义必须解决的核心问题;其次，我们综合了关于LLM供电的蜜罐的新兴文献，确定了规范架构和关键评估趋势;第三，我们绘制了蜜罐日志分析的进化路径，从简单的数据简化到自动智能生成。我们将这些发现综合成一个前瞻性的研究路线图，认为这项技术的真正潜力在于创建自主的、自我改进的欺骗系统，以应对智能的、自动化的攻击者的新兴威胁。



## **30. Practical and Stealthy Touch-Guided Jailbreak Attacks on Deployed Mobile Vision-Language Agents**

对已部署的移动视觉语言代理进行实用且隐秘的触摸引导越狱攻击 cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2510.07809v2) [paper-pdf](https://arxiv.org/pdf/2510.07809v2)

**Authors**: Renhua Ding, Xiao Yang, Zhengwei Fang, Jun Luo, Kun He, Jun Zhu

**Abstract**: Large vision-language models (LVLMs) enable autonomous mobile agents to operate smartphone user interfaces, yet vulnerabilities in their perception and interaction remain critically understudied. Existing research often relies on conspicuous overlays, elevated permissions, or unrealistic threat assumptions, limiting stealth and real-world feasibility. In this paper, we introduce a practical and stealthy jailbreak attack framework, which comprises three key components: (i) non-privileged perception compromise, which injects visual payloads into the application interface without requiring elevated system permissions; (ii) agent-attributable activation, which leverages input attribution signals to distinguish agent from human interactions and limits prompt exposure to transient intervals to preserve stealth from end users; and (iii) efficient one-shot jailbreak, a heuristic iterative deepening search algorithm (HG-IDA*) that performs keyword-level detoxification to bypass built-in safety alignment of LVLMs. Moreover, we developed three representative Android applications and curated a prompt-injection dataset for mobile agents. We evaluated our attack across multiple LVLM backends, including closed-source services and representative open-source models, and observed high planning and execution hijack rates (e.g., GPT-4o: 82.5% planning / 75.0% execution), exposing a fundamental security vulnerability in current mobile agents and underscoring critical implications for autonomous smartphone operation.

摘要: 大型视觉语言模型（LVLM）使自主移动代理能够操作智能手机用户界面，但它们的感知和交互中的漏洞仍然严重缺乏研究。现有的研究通常依赖于明显的叠加、较高的权限或不切实际的威胁假设，从而限制了隐形和现实世界的可行性。本文中，我们介绍了一个实用且隐蔽的越狱攻击框架，该框架由三个关键组件组成：（i）非特权感知妥协，它将视觉有效负载注入到应用程序界面中，而不需要提高系统权限;（ii）代理归因于激活，它利用输入归因信号将代理与人类互动区分开来，并限制即时暴露在瞬时间隔中以保持隐形最终用户;和（iii）高效的一次性越狱，这是一种启发式迭代深化搜索算法（HG-IDA*），可执行关键字级解毒以绕过LVLM的内置安全对齐。此外，我们开发了三个代表性的Android应用程序，并为移动代理策划了预算注入数据集。我们评估了多个LVLM后台（包括闭源服务和代表性开源模型）的攻击，并观察到很高的规划和执行劫持率（例如，GPT-4o：82.5%规划/ 75.0%执行），暴露了当前移动代理中的根本安全漏洞，并强调了对自主智能手机操作的关键影响。



## **31. Jailbreaking LLMs via Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge**

通过具有针对性有毒知识的语义相关嵌套场景越狱LLM cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2510.01223v2) [paper-pdf](https://arxiv.org/pdf/2510.01223v2)

**Authors**: Ning Xu, Bo Gao, Hui Dou

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks. However, they remain exposed to jailbreak attacks, eliciting harmful responses. The nested scenario strategy has been increasingly adopted across various methods, demonstrating immense potential. Nevertheless, these methods are easily detectable due to their prominent malicious intentions. In this work, we are the first to find and systematically verify that LLMs' alignment defenses are not sensitive to nested scenarios, where these scenarios are highly semantically relevant to the queries and incorporate targeted toxic knowledge. This is a crucial yet insufficiently explored direction. Based on this, we propose RTS-Attack (Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge), an adaptive and automated framework to examine LLMs' alignment. By building scenarios highly relevant to the queries and integrating targeted toxic knowledge, RTS-Attack bypasses the alignment defenses of LLMs. Moreover, the jailbreak prompts generated by RTS-Attack are free from harmful queries, leading to outstanding concealment. Extensive experiments demonstrate that RTS-Attack exhibits superior performance in both efficiency and universality compared to the baselines across diverse advanced LLMs, including GPT-4o, Llama3-70b, and Gemini-pro. Our complete code is available at https://github.com/nercode/Work. WARNING: THIS PAPER CONTAINS POTENTIALLY HARMFUL CONTENT.

摘要: 大型语言模型（LLM）在各种任务中表现出了非凡的能力。然而，他们仍然面临越狱攻击，引发有害反应。嵌套场景策略越来越多地被各种方法采用，展现出巨大的潜力。然而，这些方法由于其明显的恶意意图而很容易被检测到。在这项工作中，我们是第一个发现并系统地验证LLM的对齐防御对嵌套场景不敏感的人，这些场景与查询在语义上高度相关，并包含有针对性的有毒知识。这是一个至关重要但尚未充分探索的方向。基于此，我们提出了RTS-Attack（具有目标有毒知识的语义相关嵌套场景），这是一个自适应的自动化框架，用于检查LLM的一致性。通过构建与查询高度相关的场景并集成有针对性的有毒知识，RTS-Attack绕过了LLM的对齐防御。此外，RTS-Attack生成的越狱提示没有有害查询，具有出色的隐蔽性。大量实验表明，与GPT-4 o、Llama 3 - 70 b和Gemini-pro等各种高级LLM的基线相比，RTS-Attack在效率和通用性方面都表现出卓越的性能。我们的完整代码可在https://github.com/nercode/Work上获取。警告：本文包含潜在有害内容。



## **32. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

NeuronStrike：对对齐的LLM的神经元级攻击 cs.CR

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2509.11864v2) [paper-pdf](https://arxiv.org/pdf/2509.11864v2)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.

摘要: 安全一致对于大型语言模型（LLM）的道德部署至关重要，指导它们避免生成有害或不道德的内容。当前的对齐技术，例如有监督的微调和来自人类反馈的强化学习，仍然很脆弱，可以被精心设计的对抗提示绕过。不幸的是，此类攻击依赖于试错，缺乏跨模型的通用性，并且受到可扩展性和可靠性的限制。   本文介绍了NeuroStrike，这是一种新颖且可推广的攻击框架，它利用了对齐技术引入的一个基本漏洞：依赖于负责检测和抑制有害输入的稀疏、专门的安全神经元。我们将NeuroStrike应用于白盒和黑盒设置：在白盒设置中，NeuroStrike通过反馈激活分析识别安全神经元，并在推理期间修剪它们以禁用安全机制。在黑匣子环境中，我们提出了第一次LLM剖析攻击，该攻击通过在开权重代理模型上训练对抗提示生成器，然后将它们部署到黑匣子和专有目标上来利用安全神经元的可移植性。我们对来自主要LLM开发商的20多个开量级LLM进行了评估。通过删除目标层中不到0.6%的神经元，NeuroStrike仅使用普通恶意提示即可实现76.9%的平均攻击成功率（ASB）。此外，Neurostrike将四种多模式LLM推广到对不安全图像输入具有100%的ASB。安全神经元在架构之间有效转移，使11个微调模型的ASB达到78.5%，5个提炼模型的ASB达到77.7%。黑匣子LLM分析攻击在包括Google Gemini系列在内的五种黑匣子型号中实现了63.7%的平均ASB。



## **33. Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees**

使用结构化攻击树的LLM驱动渗透测试中的引导推理 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2509.07939v2) [paper-pdf](https://arxiv.org/pdf/2509.07939v2)

**Authors**: Katsuaki Nakano, Reza Fayyazi, Shanchieh Jay Yang, Michael Zuzak

**Abstract**: Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments

摘要: 大型语言模型（LLM）的最新进展激发了人们对自动化网络安全渗透测试工作流程的兴趣，为企业系统提供更快、更一致的漏洞评估。现有的用于渗透测试的LLM代理主要依赖于自我引导推理，这可能会产生不准确或幻觉的程序步骤。因此，LLM代理可能会采取非生产性的行动，例如利用未使用的软件库或生成重复先前策略的周期性响应。在这项工作中，我们提出了一个用于渗透测试LLM代理的引导推理管道，该管道结合了从MITRE ATT & CK矩阵（一个经过验证的渗透测试kll链）构建的确定性任务树，以将LLM的重组过程限制为明确定义的策略、技术和程序。这将推理锚定在经过验证的渗透测试方法中，并通过引导代理走向更有成效的攻击程序来过滤无效操作。为了评估我们的方法，我们使用三个LLM（Llama-3-8B、Gemini-1.5和GPT-4）构建了一个自动渗透测试LLM代理，并将其应用于导航10个HackTheBox网络安全演习，其中包含103个代表现实世界网络攻击场景的离散子任务。我们提出的推理管道使用Llama-3-8B、Gemini-1.5和GPT-4分别引导LLM代理完成71.8%、72.8%和78.6%的子任务。相比之下，使用自我引导推理的最先进的LLM渗透测试工具仅完成了13.5%、16.5%和75.7%的子任务，并且需要86.2%、118.7%和205.9%的模型查询。这表明将确定性任务树纳入LLM推理管道可以提高自动化网络安全评估的准确性和效率



## **34. PromptCOS: Towards Content-only System Prompt Copyright Auditing for LLMs**

Observtcos：面向LLM的纯内容系统提示版权审计 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2509.03117v2) [paper-pdf](https://arxiv.org/pdf/2509.03117v2)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Enhao Huang, Shuo Shao, Yuyi Wang, Zhibo Wang, Dacheng Tao, Zhan Qin

**Abstract**: System prompts are critical for shaping the behavior and output quality of large language model (LLM)-based applications, driving substantial investment in optimizing high-quality prompts beyond traditional handcrafted designs. However, as system prompts become valuable intellectual property, they are increasingly vulnerable to prompt theft and unauthorized use, highlighting the urgent need for effective copyright auditing, especially watermarking. Existing methods rely on verifying subtle logit distribution shifts triggered by a query. We observe that this logit-dependent verification framework is impractical in real-world content-only settings, primarily because (1) random sampling makes content-level generation unstable for verification, and (2) stronger instructions needed for content-level signals compromise prompt fidelity.   To overcome these challenges, we propose PromptCOS, the first content-only system prompt copyright auditing method based on content-level output similarity. PromptCOS achieves watermark stability by designing a cyclic output signal as the conditional instruction's target. It preserves prompt fidelity by injecting a small set of auxiliary tokens to encode the watermark, leaving the main prompt untouched. Furthermore, to ensure robustness against malicious removal, we optimize cover tokens, i.e., critical tokens in the original prompt, to ensure that removing auxiliary tokens causes severe performance degradation. Experimental results show that PromptCOS achieves high effectiveness (99.3% average watermark similarity), strong distinctiveness (60.8% higher than the best baseline), high fidelity (accuracy degradation no greater than 0.6%), robustness (resilience against four potential attack categories), and high computational efficiency (up to 98.1% cost saving). Our code is available at GitHub (https://github.com/LianPing-cyber/PromptCOS).

摘要: 系统提示对于塑造基于大型语言模型（LLM）的应用程序的行为和输出质量至关重要，推动了对传统手工设计之外的高质量提示进行大量投资。然而，随着系统提示成为宝贵的知识产权，它们越来越容易被及时盗窃和未经授权使用，这凸显了对有效版权审计的迫切需要，尤其是水印。现有的方法依赖于验证查询触发的微妙logit分布变化。我们观察到，这种依赖于逻辑的验证框架在现实世界的仅内容设置中是不切实际的，主要是因为（1）随机采样使得内容级生成对于验证来说不稳定，以及（2）内容级信号所需的更强指令会损害即时保真度。   为了克服这些挑战，我们提出了Inbox cos，这是第一个基于内容级输出相似性的纯内容系统提示版权审计方法。Intrutcos通过设计循环输出信号作为条件指令的目标来实现水印稳定性。它通过注入一小组辅助令牌来编码水印来保留提示的保真度，而不影响主提示。此外，为了确保针对恶意删除的鲁棒性，我们优化了掩护令牌，即原始提示中的关键令牌，以确保删除辅助令牌会导致严重的性能下降。实验结果表明，Intrutcos具有高有效性（平均水印相似度为99.3%）、强区别性（比最佳基线高60.8%）、高保真度（准确率下降不超过0.6%）、鲁棒性（对四种潜在攻击类别的弹性）和高计算效率（节省成本高达98.1%）。我们的代码可在GitHub上获取（https：//github.com/LianPing-cyber/Journaltcos）。



## **35. SoK: Exposing the Generation and Detection Gaps in LLM-Generated Phishing Through Examination of Generation Methods, Content Characteristics, and Countermeasures**

SoK：通过检查生成方法、内容特征和对策来暴露LLM生成的网络钓鱼中的生成和检测差距 cs.CR

18 pages, 5 tables, 4 figures

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2508.21457v2) [paper-pdf](https://arxiv.org/pdf/2508.21457v2)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Phishing campaigns involve adversaries masquerading as trusted vendors trying to trigger user behavior that enables them to exfiltrate private data. While URLs are an important part of phishing campaigns, communicative elements like text and images are central in triggering the required user behavior. Further, due to advances in phishing detection, attackers react by scaling campaigns to larger numbers and diversifying and personalizing content. In addition to established mechanisms, such as template-based generation, large language models (LLMs) can be used for phishing content generation, enabling attacks to scale in minutes, challenging existing phishing detection paradigms through personalized content, stealthy explicit phishing keywords, and dynamic adaptation to diverse attack scenarios. Countering these dynamically changing attack campaigns requires a comprehensive understanding of the complex LLM-related threat landscape. Existing studies are fragmented and focus on specific areas. In this work, we provide the first holistic examination of LLM-generated phishing content. First, to trace the exploitation pathways of LLMs for phishing content generation, we adopt a modular taxonomy documenting nine stages by which adversaries breach LLM safety guardrails. We then characterize how LLM-generated phishing manifests as threats, revealing that it evades detectors while emphasizing human cognitive manipulation. Third, by taxonomizing defense techniques aligned with generation methods, we expose a critical asymmetry that offensive mechanisms adapt dynamically to attack scenarios, whereas defensive strategies remain static and reactive. Finally, based on a thorough analysis of the existing literature, we highlight insights and gaps and suggest a roadmap for understanding and countering LLM-driven phishing at scale.

摘要: 网络钓鱼活动涉及伪装成值得信赖的供应商的对手，试图触发用户行为，使他们能够泄露私人数据。虽然URL是网络钓鱼活动的重要组成部分，但文本和图像等通信元素是触发所需用户行为的核心。此外，由于网络钓鱼检测的进步，攻击者通过将活动规模扩大到更大的数量以及使内容多样化和个性化来做出反应。除了基于模板的生成等已建立的机制外，大型语言模型（LLM）还可用于网络钓鱼内容生成，使攻击在几分钟内扩展，通过个性化内容、隐形显式网络钓鱼关键词和动态适应各种攻击场景来挑战现有的网络钓鱼检测范式。应对这些动态变化的攻击活动需要全面了解复杂的LLM相关威胁格局。现有的研究是零散的，并且集中在特定领域。在这项工作中，我们对LLM生成的网络钓鱼内容进行了首次全面检查。首先，为了追踪LLM用于网络钓鱼内容生成的利用途径，我们采用模块化分类法，记录对手突破LLM安全护栏的九个阶段。然后，我们描述了LLM生成的网络钓鱼如何表现为威胁，揭示了它可以逃避检测器，同时强调人类认知操纵。第三，通过对与生成方法保持一致的防御技术进行分类，我们暴露了一个关键的不对称性，即攻击机制动态地适应攻击场景，而防御策略保持静态和反应性。最后，根据对现有文献的彻底分析，我们强调了见解和差距，并提出了大规模理解和打击LLM驱动的网络钓鱼的路线图。



## **36. Failures to Surface Harmful Contents in Video Large Language Models**

未能在视频大语言模型中暴露有害内容 cs.MM

12 pages, 8 figures. Accepted to AAAI 2026

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2508.10974v2) [paper-pdf](https://arxiv.org/pdf/2508.10974v2)

**Authors**: Yuxin Cao, Wei Song, Derui Wang, Jingling Xue, Jin Song Dong

**Abstract**: Video Large Language Models (VideoLLMs) are increasingly deployed on numerous critical applications, where users rely on auto-generated summaries while casually skimming the video stream. We show that this interaction hides a critical safety gap: if harmful content is embedded in a video, either as full-frame inserts or as small corner patches, state-of-the-art VideoLLMs rarely mention the harmful content in the output, despite its clear visibility to human viewers. A root-cause analysis reveals three compounding design flaws: (1) insufficient temporal coverage resulting from the sparse, uniformly spaced frame sampling used by most leading VideoLLMs, (2) spatial information loss introduced by aggressive token downsampling within sampled frames, and (3) encoder-decoder disconnection, whereby visual cues are only weakly utilized during text generation. Leveraging these insights, we craft three zero-query black-box attacks, aligning with these flaws in the processing pipeline. Our large-scale evaluation across five leading VideoLLMs shows that the harmfulness omission rate exceeds 90% in most cases. Even when harmful content is clearly present in all frames, these models consistently fail to identify it. These results underscore a fundamental vulnerability in current VideoLLMs' designs and highlight the urgent need for sampling strategies, token compression, and decoding mechanisms that guarantee semantic coverage rather than speed alone.

摘要: 视频大型语言模型（VideoLLM）越来越多地部署在许多关键应用程序上，其中用户依赖自动生成的摘要，同时随意浏览视频流。我们表明，这种交互隐藏着一个关键的安全差距：如果有害内容嵌入视频中，无论是作为全帧插入还是作为小角补丁，那么最先进的VideoLLM很少在输出中提及有害内容，尽管它对人类观众来说是清晰可见的。根本原因分析揭示了三个复合设计缺陷：（1）大多数领先的VideoLLM使用的稀疏、均匀间隔的帧采样导致的时间覆盖不足，（2）采样帧内的激进令牌下采样引入的空间信息丢失，以及（3）编码器-解码器断开连接，从而视觉线索在文本生成过程中仅被微弱地利用。利用这些见解，我们设计了三种零查询黑匣子攻击，以与处理管道中的这些缺陷保持一致。我们对五家领先的VideoLLM进行的大规模评估显示，在大多数情况下，危害性遗漏率超过90%。即使有害内容明显存在于所有帧中，这些模型仍然无法识别它。这些结果强调了当前VideoLLM设计中的一个根本漏洞，并强调了对保证语义覆盖而不仅仅是速度的采样策略、令牌压缩和解码机制的迫切需要。



## **37. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models**

学习在大型视觉语言模型中检测未知越狱攻击 cs.CR

16 pages; Previously this version appeared as arXiv:2510.15430 which was submitted as a new work by accident

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2508.09201v3) [paper-pdf](https://arxiv.org/pdf/2508.09201v3)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.

摘要: 尽管做出了广泛的协调努力，大型视觉语言模型（LVLM）仍然容易受到越狱攻击，从而构成严重的安全风险。为了解决这个问题，现有的检测方法要么学习特定于攻击的参数，这阻碍了对不可见攻击的概括，要么依赖于数学上合理的原则，这限制了准确性和效率。为了克服这些局限性，我们提出了学习检测（Lo），这是一个通用框架，通过将重点从特定攻击的学习转移到特定任务的学习来准确检测未知越狱攻击。该框架包括用于面向安全的表示学习的多模式安全概念激活载体模块和用于无监督攻击分类的安全模式自动编码器模块。大量实验表明，我们的方法在提高效率的同时，对各种未知攻击实现了一致更高的AUROC检测。该代码可在https://anonymous.4open.science/r/Learning-to-Detect-51CB上获取。



## **38. Hidden in the Noise: Unveiling Backdoors in Audio LLMs Alignment through Latent Acoustic Pattern Triggers**

隐藏在噪音中：通过潜在声学模式触发器揭开音频LLM对齐中的后门 cs.SD

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2508.02175v3) [paper-pdf](https://arxiv.org/pdf/2508.02175v3)

**Authors**: Liang Lin, Miao Yu, Kaiwen Luo, Yibo Zhang, Lilan Peng, Dexian Wang, Xuehai Tang, Yuanhe Zhang, Xikang Yang, Zhenhong Zhou, Kun Wang, Yang Liu

**Abstract**: As Audio Large Language Models (ALLMs) emerge as powerful tools for speech processing, their safety implications demand urgent attention. While considerable research has explored textual and vision safety, audio's distinct characteristics present significant challenges. This paper first investigates: Is ALLM vulnerable to backdoor attacks exploiting acoustic triggers? In response to this issue, we introduce Hidden in the Noise (HIN), a novel backdoor attack framework designed to exploit subtle, audio-specific features. HIN applies acoustic modifications to raw audio waveforms, such as alterations to temporal dynamics and strategic injection of spectrally tailored noise. These changes introduce consistent patterns that an ALLM's acoustic feature encoder captures, embedding robust triggers within the audio stream. To evaluate ALLM robustness against audio-feature-based triggers, we develop the AudioSafe benchmark, assessing nine distinct risk types. Extensive experiments on AudioSafe and three established safety datasets reveal critical vulnerabilities in existing ALLMs: (I) audio features like environment noise and speech rate variations achieve over 90% average attack success rate. (II) ALLMs exhibit significant sensitivity differences across acoustic features, particularly showing minimal response to volume as a trigger, and (III) poisoned sample inclusion causes only marginal loss curve fluctuations, highlighting the attack's stealth.

摘要: 随着音频大语言模型（ALLM）成为语音处理的强大工具，其安全性影响迫切需要关注。虽然大量研究探索了文本和视觉安全，但音频的独特特征带来了重大挑战。本文首先研究：ALLM是否容易受到利用声学触发器的后门攻击？为了应对这个问题，我们引入了Hidden in the Noise（HIN），这是一种新颖的后门攻击框架，旨在利用微妙的特定音频特征。HIN对原始音频波进行声学修改，例如改变时间动态和战略性地注入频谱定制的噪音。这些变化引入了ALLM的声学特征编码器捕获的一致模式，并在音频流中嵌入稳健的触发器。为了评估ALLM针对基于音频特征的触发器的稳健性，我们开发了AudioSafe基准，评估九种不同的风险类型。对AudioSafe和三个已建立的安全数据集的广泛实验揭示了现有ALLM中的关键漏洞：（I）环境噪音和语音速率变化等音频特征实现了超过90%的平均攻击成功率。(II)ALLMS在声学特征中表现出显着的灵敏度差异，特别是对作为触发器的体积的响应最小，并且（III）中毒样本包含物仅引起边际损失曲线波动，凸显了攻击的隐秘性。



## **39. AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection**

AgentArmor：对Agent DeliverTrace执行程序分析以防止即时注入 cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2508.01249v3) [paper-pdf](https://arxiv.org/pdf/2508.01249v3)

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu

**Abstract**: Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's runtime traces as graph-based intermediate representations with control and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools \& data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis for sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can reduce the ASR to 3\%, with the utility drop only 1\%.

摘要: 大型语言模型（LLM）代理通过将自然语言推理与外部工具的执行相结合，提供了一个强大的新范式来解决各种问题。然而，它们的动态和不透明行为会带来严重的安全风险，特别是在存在即时注入攻击的情况下。在这项工作中，我们提出了一种新颖的见解，将代理运行时跟踪视为具有可分析语义的结构化程序。因此，我们提出了AgentArmor，这是一个程序分析框架，它将代理跟踪转换为基于图形中间表示的结构化程序依赖性表示（例如，CGM、DFG和PDG）并通过类型系统强制执行安全策略。AgentArmor由三个关键组件组成：（1）一个图形构造器，将代理的运行时跟踪重建为基于图形的中间表示，其中描述了控制和数据流;（2）一个属性注册表，附加交互工具和数据的安全相关元数据，以及（3）一个类型系统，执行静态推断和检查中间表示。通过将代理行为表示为结构化程序，AgentArmor可以对敏感数据流、信任边界和策略违规进行程序分析。我们在AgentDojo基准测试上对AgentArmor进行了评估，结果表明AgentArmor可以将ASB降低至3%，而实用程序仅下降1%。



## **40. Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs**

利用协同认知偏见来绕过LLC的安全性 cs.CL

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2507.22564v2) [paper-pdf](https://arxiv.org/pdf/2507.22564v2)

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.

摘要: 大型语言模型（LLM）在广泛的任务中表现出令人印象深刻的能力，但它们的安全机制仍然容易受到利用认知偏见（系统性偏离理性判断）的对抗攻击。与之前专注于即时工程或算法操纵的越狱方法不同，这项工作强调了多偏差相互作用在破坏LLM保障措施方面被忽视的力量。我们提出了CognitiveAttack，这是一种新型的红色团队框架，可以系统地利用个人和组合的认知偏见。通过集成有监督的微调和强化学习，CognitiveAttack生成嵌入优化的偏差组合的提示，有效地绕过安全协议，同时保持高攻击成功率。实验结果揭示了30种不同的LLM存在重大漏洞，特别是在开源模型中。与SOTA黑匣子方法PAP相比，CognitiveAttack的攻击成功率高得多（60.1% vs 31.6%），暴露了当前防御机制的严重局限性。这些发现凸显了多偏见相互作用是一种强大但未充分探索的攻击载体。这项工作通过连接认知科学和LLM安全性，引入了一种新颖的跨学科视角，为更强大、更人性化的人工智能系统铺平了道路。



## **41. LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge**

LLM无法可靠地判断（还吗？）：法学硕士作为法官稳健性的综合评估 cs.CR

**SubmitDate**: 2025-11-16    [abs](http://arxiv.org/abs/2506.09443v2) [paper-pdf](https://arxiv.org/pdf/2506.09443v2)

**Authors**: Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, Shouling Ji

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks, driving the development and widespread adoption of LLM-as-a-Judge systems for automated evaluation, including red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising critical concerns about their robustness and trustworthiness. Existing evaluation methods for LLM-based judges are often fragmented and lack a unified framework for comprehensive robustness assessment. Furthermore, the impact of prompt template design and model selection on judge robustness has rarely been explored, and their performance in real-world deployments remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. Specifically, RobustJudge investigates the effectiveness of 15 attack methods and 7 defense strategies across 12 models (RQ1), examines the impact of prompt template design and model selection (RQ2), and evaluates the security of real-world deployments (RQ3). Our study yields three key findings: (1) LLM-as-a-Judge systems are highly vulnerable to attacks such as PAIR and combined attacks, while defense mechanisms such as re-tokenization and LLM-based detectors can provide enhanced protection; (2) robustness varies substantially across prompt templates (up to 40%); (3) deploying RobustJudge on Alibaba's PAI platform uncovers previously undiscovered vulnerabilities. These results offer practical insights for building trustworthy LLM-as-a-Judge systems.

摘要: 大型语言模型（LLM）在不同任务中表现出了卓越的能力，推动了LLM作为法官自动评估系统的开发和广泛采用，包括红色团队和基准测试。然而，这些系统很容易受到对抗攻击，这些攻击可以操纵评估结果，从而引发对其稳健性和可信性的严重担忧。基于LLM的法官的现有评估方法往往支离破碎，缺乏全面稳健性评估的统一框架。此外，人们很少探讨即时模板设计和模型选择对判断稳健性的影响，而且它们在现实世界部署中的性能在很大程度上仍然未经验证。为了解决这些差距，我们引入了RobustJudge，这是一个全自动化和可扩展的框架，旨在系统性评估法学硕士即法官系统的稳健性。具体来说，RobustJudge调查了12个模型中15种攻击方法和7种防御策略的有效性（RJ 1），检查了即时模板设计和模型选择的影响（RJ 2），并评估现实世界部署的安全性（RJ 3）。我们的研究得出了三个关键发现：（1）LLM as-a-Judge系统极易受到PAIR和组合攻击等攻击，而重标记化和基于LLM的检测器等防御机制可以提供增强的保护;（2）不同提示模板的稳健性差异很大（高达40%）;（3）在阿里巴巴的PRI平台上部署RobustJudge发现了之前未发现的漏洞。这些结果为构建值得信赖的法学硕士作为法官系统提供了实用见解。



## **42. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

MCA-Bench：评估CAPTCHA针对基于VLM的攻击的稳健性的多模式基准 cs.CV

we update the paper supplement

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2506.05982v6) [paper-pdf](https://arxiv.org/pdf/2506.05982v6)

**Authors**: Zonglin Wu, Yule Xue, Yaoyao Feng, Xiaolong Wang, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.

摘要: 随着自动攻击技术的迅速发展，验证码仍然是针对恶意机器人的重要防御机制。然而，现有的CAPTCHA方案涵盖了多种形式--从静态扭曲文本和模糊图像到交互式点击、滑动谜题和基于逻辑的问题--但社区仍然缺乏统一的、大规模的、多模式基准来严格评估其安全稳健性。为了解决这一差距，我们引入了MCA-Bench，这是一个全面且可重复的基准测试套件，可将异类CAPTCHA类型集成到单个评估协议中。利用共享的视觉语言模型主干，我们为每个CAPTCHA类别微调专门的破解剂，实现一致的跨模式评估。大量实验表明，MCA-Bench有效地绘制了现代CAPTCHA设计在不同攻击环境下的脆弱性谱，并且至关重要地提供了挑战复杂性、交互深度和模型可解性如何相互关联的首次定量分析。基于这些发现，我们提出了三项可操作的设计原则，并确定了关键的开放挑战，为系统性CAPTCHA强化、公平的基准测试和更广泛的社区合作奠定了基础。数据集和代码可在线获取。



## **43. Use as Many Surrogates as You Want: Selective Ensemble Attack to Unleash Transferability without Sacrificing Resource Efficiency**

使用尽可能多的代理人：选择性发起攻击以释放可转让性，而不牺牲资源效率 cs.CV

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2505.12644v2) [paper-pdf](https://arxiv.org/pdf/2505.12644v2)

**Authors**: Bo Yang, Hengwei Zhang, Jindong Wang, Yuchen Ren, Chenhao Lin, Chao Shen, Zhengyu Zhao

**Abstract**: In surrogate ensemble attacks, using more surrogate models yields higher transferability but lower resource efficiency. This practical trade-off between transferability and efficiency has largely limited existing attacks despite many pre-trained models are easily accessible online. In this paper, we argue that such a trade-off is caused by an unnecessary common assumption, i.e., all models should be \textit{identical} across iterations. By lifting this assumption, we can use as many surrogates as we want to unleash transferability without sacrificing efficiency. Concretely, we propose Selective Ensemble Attack (SEA), which dynamically selects diverse models (from easily accessible pre-trained models) across iterations based on our new interpretation of decoupling within-iteration and cross-iteration model diversity. In this way, the number of within-iteration models is fixed for maintaining efficiency, while only cross-iteration model diversity is increased for higher transferability. Experiments on ImageNet demonstrate the superiority of SEA in various scenarios. For example, when dynamically selecting 4 from 20 accessible models, SEA yields 8.5% higher transferability than existing attacks under the same efficiency. The superiority of SEA also generalizes to real-world systems, such as commercial vision APIs and large vision-language models. Overall, SEA opens up the possibility of adaptively balancing transferability and efficiency according to specific resource requirements.

摘要: 在代理集成攻击中，使用更多代理模型会产生更高的可移植性，但资源效率较低。尽管许多预先训练的模型可以轻松在线访问，但可移植性和效率之间的这种实际权衡在很大程度上限制了现有的攻击。在本文中，我们认为这种权衡是由不必要的共同假设引起的，即，所有模型在迭代中都应该\textit{equivalent}。通过取消这一假设，我们可以使用尽可能多的代理人，以释放可转移性，而不牺牲效率。具体来说，我们提出了选择性集合攻击（SEA），它基于我们对迭代内脱钩和跨迭代模型多样性的新解释，在迭代中动态选择不同的模型（从易于访问的预训练模型中）。通过这种方式，迭代内模型的数量是固定的，以保持效率，而仅增加交叉迭代模型的多样性以获得更高的可移植性。ImageNet上的实验证明了SEA在各种场景下的优越性。例如，当从20个可访问模型中动态选择4个时，在相同效率下，SEA的可转移性比现有攻击高出8.5%。SEA的优势还推广到现实世界的系统，例如商业视觉API和大型视觉语言模型。总体而言，SEA开辟了根据特定资源要求自适应地平衡可转移性和效率的可能性。



## **44. Revisiting Model Inversion Evaluation: From Misleading Standards to Reliable Privacy Assessment**

重温模型倒置评估：从误导性标准到可靠的隐私评估 cs.LG

To support future work, we release our MLLM-based MI evaluation framework and benchmarking suite at https://github.com/hosytuyen/MI-Eval-MLLM

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2505.03519v4) [paper-pdf](https://arxiv.org/pdf/2505.03519v4)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information from private training data by exploiting access to machine learning models T. To evaluate such attacks, the standard evaluation framework relies on an evaluation model E, trained under the same task design as T. This framework has become the de facto standard for assessing progress in MI research, used across nearly all recent MI studies without question. In this paper, we present the first in-depth study of this evaluation framework. In particular, we identify a critical issue of this standard framework: Type-I adversarial examples. These are reconstructions that do not capture the visual features of private training data, yet are still deemed successful by T and ultimately transferable to E. Such false positives undermine the reliability of the standard MI evaluation framework. To address this issue, we introduce a new MI evaluation framework that replaces the evaluation model E with advanced Multimodal Large Language Models (MLLMs). By leveraging their general-purpose visual understanding, our MLLM-based framework does not depend on training of shared task design as in T, thus reducing Type-I transferability and providing more faithful assessments of reconstruction success. Using our MLLM-based evaluation framework, we reevaluate 27 diverse MI attack setups and empirically reveal consistently high false positive rates under the standard evaluation framework. Importantly, we demonstrate that many state-of-the-art (SOTA) MI methods report inflated attack accuracy, indicating that actual privacy leakage is significantly lower than previously believed. By uncovering this critical issue and proposing a robust solution, our work enables a reassessment of progress in MI research and sets a new standard for reliable and robust evaluation. Code can be found in https://github.com/hosytuyen/MI-Eval-MLLM

摘要: 模型倒置（MI）攻击旨在通过利用对机器学习模型T的访问来从私人训练数据中重建信息。为了评估此类攻击，标准评估框架依赖于评估模型E，该模型在与T相同的任务设计下训练。该框架已成为评估MI研究进展的事实标准，几乎所有最近的MI研究都毫无疑问地使用了该框架。在本文中，我们对该评估框架进行了首次深入研究。特别是，我们确定了这个标准框架的一个关键问题：I型对抗性示例。这些重建并没有捕捉到私人训练数据的视觉特征，但仍然被T认为是成功的，并最终转移到E。这种假阳性损害了标准管理信息评价框架的可靠性。为了解决这个问题，我们引入了一个新的MI评估框架，用先进的多模态大型语言模型（MLLM）取代了评估模型E。通过利用他们的通用视觉理解，我们基于MLLM的框架不依赖于T中的共享任务设计的训练，从而降低了I型可转移性，并提供了更忠实的重建成功评估。使用我们基于MLLM的评估框架，我们重新评估了27种不同的MI攻击设置，并根据经验揭示了标准评估框架下一贯的高误报率。重要的是，我们证明了许多最先进的（SOTA）MI方法报告了夸大的攻击准确性，这表明实际的隐私泄露显着低于以前认为的。通过揭示这一关键问题并提出一个强有力的解决方案，我们的工作能够重新评估MI研究的进展，并为可靠和强大的评估设定了新的标准。代码可在https://github.com/hosytuyen/MI-Eval-MLLM找到



## **45. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

一张图片即可：用单个图像毒害视觉文档检索增强生成 cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2504.02132v3) [paper-pdf](https://arxiv.org/pdf/2504.02132v3)

**Authors**: Ezzeldin Shereen, Dan Ristea, Shae McFadden, Burak Hasircioglu, Vasilios Mavroudis, Chris Hicks

**Abstract**: Retrieval-augmented generation (RAG) is instrumental for inhibiting hallucinations in large language models (LLMs) through the use of a factual knowledge base (KB). Although PDF documents are prominent sources of knowledge, text-based RAG pipelines are ineffective at capturing their rich multi-modal information. In contrast, visual document RAG (VD-RAG) uses screenshots of document pages as the KB, which has been shown to achieve state-of-the-art results. However, by introducing the image modality, VD-RAG introduces new attack vectors for adversaries to disrupt the system by injecting malicious documents into the KB. In this paper, we demonstrate the vulnerability of VD-RAG to poisoning attacks targeting both retrieval and generation. We define two attack objectives and demonstrate that both can be realized by injecting only a single adversarial image into the KB. Firstly, we introduce a targeted attack against one or a group of queries with the goal of spreading targeted disinformation. Secondly, we present a universal attack that, for any potential user query, influences the response to cause a denial-of-service in the VD-RAG system. We investigate the two attack objectives under both white-box and black-box assumptions, employing a multi-objective gradient-based optimization approach as well as prompting state-of-the-art generative models. Using two visual document datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (vision language models), we show VD-RAG is vulnerable to poisoning attacks in both the targeted and universal settings, yet demonstrating robustness to black-box attacks in the universal setting.

摘要: 检索增强生成（RAG）有助于通过使用事实知识库（KB）来抑制大型语言模型（LLM）中的幻觉。尽管PDF文档是重要的知识来源，但基于文本的RAG管道在捕获其丰富的多模式信息方面效率低下。相比之下，视觉文档RAG（VD-RAG）使用文档页面的屏幕截图作为KB，已被证明可以实现最先进的结果。然而，通过引入图像模式，VD-RAG为对手引入了新的攻击载体，通过将恶意文档注入知识库来破坏系统。在本文中，我们展示了VD-RAG对针对检索和生成的中毒攻击的脆弱性。我们定义了两个攻击目标，并证明这两个目标都可以通过仅向知识库中注入单个对抗图像来实现。首先，我们对一个或一组查询引入有针对性的攻击，目标是传播有针对性的虚假信息。其次，我们提出了一种通用攻击，对于任何潜在的用户查询，该攻击都会影响响应，从而导致VD-RAG系统中的拒绝服务。我们调查的两个攻击目标下的白盒和黑盒的假设，采用多目标的基于梯度的优化方法，以及促使国家的最先进的生成模型。使用两个可视化文档数据集，一组不同的最先进的检索器（嵌入模型）和生成器（视觉语言模型），我们表明VD-RAG在目标和通用设置中都容易受到中毒攻击，但在通用设置中表现出对黑盒攻击的鲁棒性。



## **46. LightDefense: A Lightweight Uncertainty-Driven Defense against Jailbreaks via Shifted Token Distribution**

LightDefense：通过转移代币分发针对越狱的轻量级不确定性驱动防御 cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2504.01533v2) [paper-pdf](https://arxiv.org/pdf/2504.01533v2)

**Authors**: Zhuoran Yang, Yanyong Zhang

**Abstract**: Large Language Models (LLMs) face threats from jailbreak prompts. Existing methods for defending against jailbreak attacks are primarily based on auxiliary models. These strategies, however, often require extensive data collection or training. We propose LightDefense, a lightweight defense mechanism targeted at white-box models, which utilizes a safety-oriented direction to adjust the probabilities of tokens in the vocabulary, making safety disclaimers appear among the top tokens after sorting tokens by probability in descending order. We further innovatively leverage LLM's uncertainty about prompts to measure their harmfulness and adaptively adjust defense strength, effectively balancing safety and helpfulness. The effectiveness of LightDefense in defending against 5 attack methods across 2 target LLMs, without compromising helpfulness to benign user queries, highlights its potential as a novel and lightweight defense mechanism, enhancing security of LLMs.

摘要: 大型语言模型（LLM）面临来自越狱提示的威胁。现有的针对越狱攻击的防御方法主要基于辅助模型。然而，这些战略往往需要广泛的数据收集或培训。我们提出LightDefense，这是一种针对白盒模型的轻量级防御机制，利用以安全为导向的方向来调整词汇表中代币的概率，使安全免责声明在按概率降序排序后出现在前几名代币中。我们进一步创新性地利用LLM对提示的不确定性来衡量其危害性，并自适应地调整防御强度，有效地平衡了安全性和有益性。LightDefense在2个目标LLM上防御5种攻击方法的有效性，而不影响对良性用户查询的帮助，突出了其作为一种新型轻量级防御机制的潜力，增强了LLM的安全性。



## **47. IPAD: Inverse Prompt for AI Detection - A Robust and Interpretable LLM-Generated Text Detector**

iPad：人工智能检测的反向提示-一个强大且可解释的LLM生成文本检测器 cs.LG

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2502.15902v3) [paper-pdf](https://arxiv.org/pdf/2502.15902v3)

**Authors**: Zheng Chen, Yushi Feng, Jisheng Dang, Yue Deng, Changyang He, Hongxi Pu, Haoxuan Li, Bo Li

**Abstract**: Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinguishing between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.

摘要: 大型语言模型（LLM）在文本生成方面已经达到了人类水平的流畅性，这使得区分人类书面文本和LLM生成的文本变得复杂。这增加了误用的风险，并突出了对可靠探测器的需求。然而，现有的检测器表现出对分布外（OOD）数据和攻击数据的鲁棒性差，这对于现实世界的场景是至关重要的。此外，他们努力提供可解释的证据来支持他们的决定，从而破坏了可靠性。鉴于这些挑战，我们提出了iPad（人工智能检测反向提示），这是一个新颖的框架，由一个提示反向器和两个区分器组成，用于识别可能生成输入文本的预测提示，用于检查输入文本与预测提示对齐的可能性。经验评估表明，iPad在分销内数据上的表现比最强基线高出9.05%（平均召回），在分销外数据上的表现比最强基线高出12.93%（AUROC），在受攻击数据上的表现比最强基线高出5.48%（AUROC）。iPad还在结构化数据集上表现出色。此外，还进行了可解释性评估，以说明iPad通过允许用户直接检查决策证据来增强了人工智能检测的可信度，从而为其最先进的检测结果提供了可解释的支持。



## **48. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

大型推理模型的隐藏风险：R1的安全评估 cs.CY

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2502.12659v4) [paper-pdf](https://arxiv.org/pdf/2502.12659v4)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models (LRMs), such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source reasoning models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on open LRMs is needed. (2) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (3) Safety thinking emerges in the reasoning process of LRMs, but fails frequently against adversarial attacks. (4) The thinking process in R1 models poses greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.

摘要: OpenAI-o3和DeepSeek-R1等大型推理模型（LRM）的快速发展使复杂推理相对于非推理大型语言模型（LRM）有了显着改进。然而，它们增强的功能，加上DeepSeek-R1等模型的开源访问，引发了严重的安全问题，特别是关于它们被滥用的可能性。在这项工作中，我们对这些推理模型进行了全面的安全评估，利用既定的安全基准来评估它们对安全法规的遵守性。此外，我们还调查了它们对越狱和即时注射等对抗攻击的敏感性，以评估它们在现实应用中的稳健性。通过多方面的分析，我们发现了四个关键发现：（1）开源推理模型和o3-mini模型之间在安全基准和攻击方面存在显着的安全差距，这表明需要对开放LRM做出更多的安全努力。(2)模型的推理能力越强，在回答不安全问题时可能造成的潜在危害就越大。(3)安全思维出现在LRM的推理过程中，但在对抗性攻击时经常失败。(4)R1模型中的思维过程比其最终答案带来了更大的安全问题。我们的研究为推理模型的安全性影响提供了深入的见解，并强调了进一步提高R1模型安全性以缩小差距的必要性。



## **49. What You See Is Not Always What You Get: Evaluating GPT's Comprehension of Source Code**

您所看到的并不总是您所得到的：评估GPT对源代码的理解 cs.SE

This work has been accepted at APSEC 2025

**SubmitDate**: 2025-11-17    [abs](http://arxiv.org/abs/2412.08098v3) [paper-pdf](https://arxiv.org/pdf/2412.08098v3)

**Authors**: Jiawen Wen, Bangshuo Zhu, Huaming Chen

**Abstract**: Recent studies have demonstrated outstanding capabilities of large language models (LLMs) in software engineering tasks, including code generation and comprehension. While LLMs have shown significant potential in assisting with coding, LLMs are vulnerable to adversarial attacks. In this paper, we investigate the vulnerability of LLMs to imperceptible attacks. This class of attacks manipulate source code at the character level, which renders the changes invisible to human reviewers yet effective in misleading LLMs' behaviour. We devise these attacks into four distinct categories and analyse their impacts on code analysis and comprehension tasks. These four types of imperceptible character attacks include coding reordering, invisible coding characters, code deletions, and code homoglyphs. To assess the robustness of state-of-the-art LLMs, we present a systematic evaluation across multiple models using both perturbed and clean code snippets. Two evaluation metrics, model confidence using log probabilities of response and response correctness, are introduced. The results reveal that LLMs are susceptible to imperceptible coding perturbations, with varying degrees of degradation highlighted across different LLMs. Furthermore, we observe a consistent negative correlation between perturbation magnitude and model performance. These results highlight the urgent need for robust LLMs capable of manoeuvring behaviours under imperceptible adversarial conditions.

摘要: 最近的研究证明了大型语言模型（LLM）在软件工程任务（包括代码生成和理解）中的出色能力。虽然LLM在协助编码方面表现出了巨大的潜力，但LLM很容易受到对抗攻击。在本文中，我们研究了LLM对不可感知攻击的脆弱性。这类攻击在字符级别操纵源代码，这使得更改对人类审查者来说是不可见的，但却有效误导LLM的行为。我们将这些攻击分为四个不同的类别，并分析它们对代码分析和理解任务的影响。这四种不可感知的字符攻击包括编码重新排序、隐形编码字符、代码删除和代码同字形。为了评估最先进的LLM的稳健性，我们使用扰动和干净的代码片段对多个模型进行了系统性评估。引入了两个评估指标，即使用响应的日志概率的模型置信度和响应正确性。结果表明，LLM容易受到不可察觉的编码扰动，不同LLM之间突出显示了不同程度的退化。此外，我们观察到一个一致的扰动幅度和模型性能之间的负相关。这些结果强调了迫切需要强大的LLM能够在难以察觉的对抗条件下操纵行为。



## **50. Eguard: Defending LLM Embeddings Against Inversion Attacks via Text Mutual Information Optimization**

Eguard：通过文本互信息优化保护LLM嵌入免受倒置攻击 cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2411.05034v2) [paper-pdf](https://arxiv.org/pdf/2411.05034v2)

**Authors**: Tiantian Liu, Hongwei Yao, Feng Lin, Tong Wu, Zhan Qin, Kui Ren

**Abstract**: Embeddings have become a cornerstone in the functionality of large language models (LLMs) due to their ability to transform text data into rich, dense numerical representations that capture semantic and syntactic properties. These embedding vector databases serve as the long-term memory of LLMs, enabling efficient handling of a wide range of natural language processing tasks. However, the surge in popularity of embedding vector databases in LLMs has been accompanied by significant concerns about privacy leakage. Embedding vector databases are particularly vulnerable to embedding inversion attacks, where adversaries can exploit the embeddings to reverse-engineer and extract sensitive information from the original text data. Existing defense mechanisms have shown limitations, often struggling to balance security with the performance of downstream tasks. To address these challenges, we introduce Eguard, a novel defense mechanism designed to mitigate embedding inversion attacks. Eguard employs a transformer-based projection network and text mutual information optimization to safeguard embeddings while preserving the utility of LLMs. Our approach significantly reduces privacy risks, protecting over 95% of tokens from inversion while maintaining high performance across downstream tasks consistent with original embeddings.

摘要: 嵌入已成为大型语言模型（LLM）功能的基石，因为它们能够将文本数据转换为捕获语义和语法属性的丰富、密集的数字表示。这些嵌入式载体数据库充当LLM的长期存储器，能够高效处理广泛的自然语言处理任务。然而，在LLM中嵌入载体数据库的普及率激增，同时也伴随着对隐私泄露的严重担忧。嵌入式载体数据库特别容易受到嵌入倒置攻击，对手可以利用嵌入进行反向工程并从原始文本数据中提取敏感信息。现有的防御机制已经显示出局限性，经常难以平衡安全性与下游任务的性能。为了解决这些挑战，我们引入了Eguard，这是一种新颖的防御机制，旨在减轻嵌入倒置攻击。Eguard采用基于转换器的投影网络和文本互信息优化来保护嵌入，同时保留LLM的实用性。我们的方法显着降低了隐私风险，保护超过95%的令牌免受倒置，同时在下游任务中保持与原始嵌入一致的高性能。



