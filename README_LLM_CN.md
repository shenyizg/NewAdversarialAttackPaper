# Latest Large Language Model Attack Papers
**update at 2025-07-31 10:31:07**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

不要落后，RAG：使用RAG进行免训练对抗检测 cs.AI

Accepted at VecDB @ ICML 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2504.04858v3) [paper-pdf](http://arxiv.org/pdf/2504.04858v3)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.

摘要: 对抗性补丁攻击通过嵌入误导深度模型的局部扰动对视觉系统构成重大威胁。传统的防御方法通常需要重新训练或微调，这使得它们在实际部署中不切实际。我们提出了一个免训练的视觉检索增强生成（VRAG）框架，该框架集成了视觉语言模型（VLM）用于对抗性补丁检测。通过在不断扩展的数据库中检索与存储的攻击相似的视觉上相似的补丁和图像，VRAG执行生成推理以识别不同的攻击类型，所有这些都无需额外的训练或微调。我们广泛评估了开源大型VLM，包括Qwen-VL-Plus、Qwen2.5-VL-72 B和UI-TARS-72 B-DPO，以及Gemini-2.0（一种闭源模型）。值得注意的是，开源UI-TARS-72 B-DPO模型实现了高达95%的分类准确率，为开源对抗补丁检测奠定了新的最新水平。Gemini-2.0的总体准确率达到了最高的98%，但仍然是闭源的。实验结果证明了VRAG在以最少的人类注释识别各种对抗补丁方面的有效性，为针对不断发展的对抗补丁攻击的稳健、实用的防御铺平了道路。



## **2. Efficient Differentially Private Fine-Tuning of LLMs via Reinforcement Learning**

通过强化学习对LLM进行高效的差异私有微调 cs.LG

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22565v1) [paper-pdf](http://arxiv.org/pdf/2507.22565v1)

**Authors**: Afshin Khadangi, Amir Sartipi, Igor Tchappi, Ramin Bahmani, Gilbert Fridgen

**Abstract**: The tension between data privacy and model utility has become the defining bottleneck for the practical deployment of large language models (LLMs) trained on sensitive corpora including healthcare. Differentially private stochastic gradient descent (DP-SGD) guarantees formal privacy, yet it does so at a pronounced cost: gradients are forcibly clipped and perturbed with noise, degrading sample efficiency and final accuracy. Numerous variants have been proposed to soften this trade-off, but they all share a handicap: their control knobs are hard-coded, global, and oblivious to the evolving optimization landscape. Consequently, practitioners are forced either to over-spend privacy budget in pursuit of utility, or to accept mediocre models in order to stay within privacy constraints. We present RLDP, the first framework to cast DP optimization itself as a closed-loop control problem amenable to modern deep reinforcement learning (RL). RLDP continuously senses rich statistics of the learning dynamics and acts by selecting fine-grained per parameter gradient-clipping thresholds as well as the magnitude of injected Gaussian noise. A soft actor-critic (SAC) hyper-policy is trained online during language model fine-tuning; it learns, from scratch, how to allocate the privacy budget where it matters and when it matters. Across more than 1,600 ablation experiments on GPT2-small, Llama-1B, Llama-3B, and Mistral-7B, RLDP delivers perplexity reductions of 1.3-30.5% (mean 5.4%) and an average 5.6% downstream utility gain. RLDP reaches each baseline's final utility after only 13-43% of the gradient-update budget (mean speed-up 71%), all while honoring the same ($\epsilon$, $\delta$)-DP contract and exhibiting equal or lower susceptibility to membership-inference and canary-extraction attacks.

摘要: 数据隐私和模型实用性之间的紧张关系已成为在包括医疗保健在内的敏感数据库上训练的大型语言模型（LLM）实际部署的决定性瓶颈。差异私密随机梯度下降（DP-BCD）保证了形式的隐私，但它这样做的代价是巨大的：梯度被强制剪裁并受到噪音的干扰，从而降低了样本效率和最终准确性。人们提出了许多变体来软化这种权衡，但它们都有一个共同的障碍：它们的控制旋钮是硬编码的、全球性的，并且没有注意到不断变化的优化环境。因此，从业者被迫要么过度花费隐私预算以追求实用性，要么接受平庸的模型以保持隐私限制。我们提出了RLCC，这是第一个将DP优化本身视为适合现代深度强化学习（RL）的闭环控制问题的框架。RLCC持续感知学习动态的丰富统计数据，并通过选择细粒度的每参数梯度剪裁阈值以及注入的高斯噪音的幅度来采取行动。软行动者-批评者（SAC）超策略在语言模型微调期间在线训练;它从头开始学习如何在重要的地方和何时分配隐私预算。在对GPT 2-small、Llama-1B、Llama-3B和Mistral-7 B进行的1，600多项消融实验中，RLCC可降低1.3-30.5%（平均5.4%），平均可降低5.6%的下游效用。RLCC仅在梯度更新预算的13-43%（平均加速71%）后就达到每个基线的最终效用，同时遵守相同的（$\$，$\delta$）-DP合同，并且对成员资格推断和金丝雀提取攻击的敏感性相同或更低。



## **3. Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs**

利用协同认知偏见来绕过LLC的安全性 cs.CL

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22564v1) [paper-pdf](http://arxiv.org/pdf/2507.22564v1)

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.

摘要: 大型语言模型（LLM）在广泛的任务中表现出令人印象深刻的能力，但它们的安全机制仍然容易受到利用认知偏见（系统性偏离理性判断）的对抗攻击。与之前专注于即时工程或算法操纵的越狱方法不同，这项工作强调了多偏差相互作用在破坏LLM保障措施方面被忽视的力量。我们提出了CognitiveAttack，这是一种新型的红色团队框架，可以系统地利用个人和组合的认知偏见。通过集成有监督的微调和强化学习，CognitiveAttack生成嵌入优化的偏差组合的提示，有效地绕过安全协议，同时保持高攻击成功率。实验结果揭示了30种不同的LLM存在重大漏洞，特别是在开源模型中。与SOTA黑匣子方法PAP相比，CognitiveAttack的攻击成功率高得多（60.1% vs 31.6%），暴露了当前防御机制的严重局限性。这些发现凸显了多偏见相互作用是一种强大但未充分探索的攻击载体。这项工作通过连接认知科学和LLM安全性，引入了一种新颖的跨学科视角，为更强大、更人性化的人工智能系统铺平了道路。



## **4. ETrace:Event-Driven Vulnerability Detection in Smart Contracts via LLM-Based Trace Analysis**

ETrace：通过基于LLM的跟踪分析在智能合同中进行事件驱动的漏洞检测 cs.CR

4 pages, 1 figure. To appear in Proceedings of the 16th Asia-Pacific  Symposium on Internetware (Internetware 2025), ACM ICPS. DOI:  https://doi.org/10.1145/3755881.3755934

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2506.15790v3) [paper-pdf](http://arxiv.org/pdf/2506.15790v3)

**Authors**: Chenyang Peng, Haijun Wang, Yin Wu, Hao Wu, Ming Fan, Yitao Zhao, Ting Liu

**Abstract**: With the advance application of blockchain technology in various fields, ensuring the security and stability of smart contracts has emerged as a critical challenge. Current security analysis methodologies in vulnerability detection can be categorized into static analysis and dynamic analysis methods.However, these existing traditional vulnerability detection methods predominantly rely on analyzing original contract code, not all smart contracts provide accessible code.We present ETrace, a novel event-driven vulnerability detection framework for smart contracts, which uniquely identifies potential vulnerabilities through LLM-powered trace analysis without requiring source code access. By extracting fine-grained event sequences from transaction logs, the framework leverages Large Language Models (LLMs) as adaptive semantic interpreters to reconstruct event analysis through chain-of-thought reasoning. ETrace implements pattern-matching to establish causal links between transaction behavior patterns and known attack behaviors. Furthermore, we validate the effectiveness of ETrace through preliminary experimental results.

摘要: 随着区块链技术在各个领域的深入应用，确保智能合约的安全性和稳定性已成为一项严峻的挑战。当前漏洞检测中的安全分析方法可以分为静态分析和动态分析方法。然而，这些现有的传统漏洞检测方法主要依赖于分析原始合同代码，并非所有智能合同都提供可访问代码。我们提出ETrace，一种新颖的事件驱动的智能合同漏洞检测框架，它通过LLM支持的跟踪分析来唯一地识别潜在漏洞，而无需访问源代码。通过从事务日志中提取细粒度事件序列，该框架利用大型语言模型（LLM）作为自适应语义解释器，通过思想链推理重建事件分析。ETrace实现模式匹配，以建立事务行为模式和已知攻击行为之间的因果联系。此外，我们通过初步实验结果验证了ETrace的有效性。



## **5. Invisible Injections: Exploiting Vision-Language Models Through Steganographic Prompt Embedding**

隐形注射：通过隐写提示嵌入利用视觉语言模型 cs.CR

14 Pages

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22304v1) [paper-pdf](http://arxiv.org/pdf/2507.22304v1)

**Authors**: Chetan Pathade

**Abstract**: Vision-language models (VLMs) have revolutionized multimodal AI applications but introduce novel security vulnerabilities that remain largely unexplored. We present the first comprehensive study of steganographic prompt injection attacks against VLMs, where malicious instructions are invisibly embedded within images using advanced steganographic techniques. Our approach demonstrates that current VLM architectures can inadvertently extract and execute hidden prompts during normal image processing, leading to covert behavioral manipulation. We develop a multi-domain embedding framework combining spatial, frequency, and neural steganographic methods, achieving an overall attack success rate of 24.3% (plus or minus 3.2%, 95% CI) across leading VLMs including GPT-4V, Claude, and LLaVA, with neural steganography methods reaching up to 31.8%, while maintaining reasonable visual imperceptibility (PSNR greater than 38 dB, SSIM greater than 0.94). Through systematic evaluation on 12 diverse datasets and 8 state-of-the-art models, we reveal moderate but meaningful vulnerabilities in current VLM architectures and propose effective countermeasures. Our findings have significant implications for VLM deployment in security-critical applications and highlight the need for proportionate multimodal AI security frameworks.

摘要: 视觉语言模型（VLM）彻底改变了多模式人工智能应用程序，但也引入了新的安全漏洞，这些漏洞在很大程度上尚未被探索。我们首次全面研究了针对VLM的隐写提示注入攻击，其中恶意指令使用先进的隐写技术以不可见的方式嵌入图像中。我们的方法表明，当前的VLM架构可能会在正常图像处理期间无意中提取和执行隐藏提示，从而导致隐蔽的行为操纵。我们开发了一个结合空间、频率和神经隐写方法的多域嵌入框架，总体攻击成功率达到24.3%（正负3.2%，95%CI）包括GPT-4V、Claude和LLaVA，神经隐写术方法高达31.8%，同时保持合理的视觉不可感知性（PSNR大于38分贝，SSIM大于0.94）。通过对12个不同数据集和8个最先进模型的系统评估，我们揭示了当前VLM架构中适度但有意义的漏洞，并提出了有效的对策。我们的研究结果对安全关键应用程序中的VLM部署具有重大影响，并强调了对相称的多模式人工智能安全框架的需求。



## **6. Large Language Model-Based Framework for Explainable Cyberattack Detection in Automatic Generation Control Systems**

自动发电控制系统中基于大语言模型的可解释网络攻击检测框架 cs.CR

Accepted Publication

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.22239v1) [paper-pdf](http://arxiv.org/pdf/2507.22239v1)

**Authors**: Muhammad Sharshar, Ahmad Mohammad Saber, Davor Svetinovic, Amr M. Youssef, Deepa Kundur, Ehab F. El-Saadany

**Abstract**: The increasing digitization of smart grids has improved operational efficiency but also introduced new cybersecurity vulnerabilities, such as False Data Injection Attacks (FDIAs) targeting Automatic Generation Control (AGC) systems. While machine learning (ML) and deep learning (DL) models have shown promise in detecting such attacks, their opaque decision-making limits operator trust and real-world applicability. This paper proposes a hybrid framework that integrates lightweight ML-based attack detection with natural language explanations generated by Large Language Models (LLMs). Classifiers such as LightGBM achieve up to 95.13% attack detection accuracy with only 0.004 s inference latency. Upon detecting a cyberattack, the system invokes LLMs, including GPT-3.5 Turbo, GPT-4 Turbo, and GPT-4o mini, to generate human-readable explanation of the event. Evaluated on 100 test samples, GPT-4o mini with 20-shot prompting achieved 93% accuracy in identifying the attack target, a mean absolute error of 0.075 pu in estimating attack magnitude, and 2.19 seconds mean absolute error (MAE) in estimating attack onset. These results demonstrate that the proposed framework effectively balances real-time detection with interpretable, high-fidelity explanations, addressing a critical need for actionable AI in smart grid cybersecurity.

摘要: 智能电网日益数字化提高了运营效率，但也带来了新的网络安全漏洞，例如针对自动发电控制（AGC）系统的虚假数据注入攻击（FDIA）。虽然机器学习（ML）和深度学习（DL）模型在检测此类攻击方面表现出了希望，但其不透明的决策限制了操作员的信任和现实世界的适用性。本文提出了一种混合框架，集成了轻量级的ML为基础的攻击检测与自然语言解释产生的大语言模型（LLM）。LightGBM等分类器的攻击检测准确率高达95.13%，推理延迟仅为0.004 s。检测到网络攻击后，系统会调用LLM（包括GPT-3.5涡轮、GPT-4涡轮和GPT-4 o mini）来生成人类可读的事件解释。经过100个测试样本的评估，带20发提示的GPT-4 o mini识别攻击目标的准确率达到了93%，估计攻击幅度的平均绝对误差为0.075 pu，估计攻击开始的平均绝对误差为2.19秒。这些结果表明，拟议的框架有效地平衡了实时检测与可解释的高保真解释，满足了智能电网网络安全中对可操作人工智能的迫切需求。



## **7. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

22 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2411.08003v3) [paper-pdf](http://arxiv.org/pdf/2411.08003v3)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.

摘要: 在对抗环境中（例如网络攻击和虚假信息攻击）对大型语言模型（LLM）的输出进行归因会带来重大挑战，而且其重要性可能会越来越大。我们从理论和实证的角度来处理这个归因问题，借鉴形式语言理论（极限识别）和对不断扩大的LLM生态系统的数据驱动分析。通过将LLM的一组可能输出建模为形式语言，我们分析有限的文本样本是否可以唯一地确定原始模型。我们的结果表明，在模型之间能力重叠的温和假设下，某些类别的LLM从根本上无法仅从其输出中识别。我们描绘了理论可识别性的四种制度：（1）无限一类确定性（离散）LLM语言不可识别（Gold的经典结果来自1967年）;（2）无限类概率LLM也是不可识别的（通过确定性情况的扩展）;（3）有限类确定性LLM是可识别的（与Angluin的泄密标准一致）;以及（4）即使是有限类的概率LLM也可能是不可识别的（我们提供了一个新的反例来建立这个负结果）。作为对这些理论见解的补充，我们量化了近年来给定输出的合理模型起源（假设空间）数量的爆炸式增长。即使在保守的假设下--每个开源模型最多在一个新厕所上进行微调--不同候选模型的数量也大约每0.5年翻一番，并且允许多数据集微调组合可以产生翻倍的时间短至0.28年。这种组合增长，加上所有模型和潜在用户的暴力可能性归因的非凡计算成本，使得详尽的归因在实践中不可行。



## **8. Strategic Deflection: Defending LLMs from Logit Manipulation**

战略偏转：保护LLM免受Logit操纵 cs.CR

20 pages

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.22160v1) [paper-pdf](http://arxiv.org/pdf/2507.22160v1)

**Authors**: Yassine Rachidy, Jihad Rbaiti, Youssef Hmamouche, Faissal Sehbaoui, Amal El Fallah Seghrouchni

**Abstract**: With the growing adoption of Large Language Models (LLMs) in critical areas, ensuring their security against jailbreaking attacks is paramount. While traditional defenses primarily rely on refusing malicious prompts, recent logit-level attacks have demonstrated the ability to bypass these safeguards by directly manipulating the token-selection process during generation. We introduce Strategic Deflection (SDeflection), a defense that redefines the LLM's response to such advanced attacks. Instead of outright refusal, the model produces an answer that is semantically adjacent to the user's request yet strips away the harmful intent, thereby neutralizing the attacker's harmful intent. Our experiments demonstrate that SDeflection significantly lowers Attack Success Rate (ASR) while maintaining model performance on benign queries. This work presents a critical shift in defensive strategies, moving from simple refusal to strategic content redirection to neutralize advanced threats.

摘要: 随着大型语言模型（LLM）在关键领域的日益普及，确保其安全性以防止越狱攻击至关重要。虽然传统的防御主要依赖于拒绝恶意提示，但最近的logit级攻击已经证明了通过在生成过程中直接操纵令牌选择过程来绕过这些保护措施的能力。我们引入战略偏转（SDeflection），一种重新定义LLM对这种高级攻击的反应的防御。该模型不是直接拒绝，而是产生一个在语义上与用户请求相邻但剥离有害意图的答案，从而中和攻击者的有害意图。我们的实验表明，SDeflection显着降低了攻击成功率（ASB），同时保持了良性查询的模型性能。这项工作带来了防御策略的关键转变，从简单的拒绝转向战略内容重定向以抵消高级威胁。



## **9. Prompt Optimization and Evaluation for LLM Automated Red Teaming**

LLM自动化红队编组的快速优化与评价 cs.CR

9 pages, 5 Figures, and 1 Appendix item

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.22133v1) [paper-pdf](http://arxiv.org/pdf/2507.22133v1)

**Authors**: Michael Freenor, Lauren Alvarez, Milton Leal, Lily Smith, Joel Garrett, Yelyzaveta Husieva, Madeline Woodruff, Ryan Miller, Erich Kummerfeld, Rafael Medeiros, Sander Schulhoff

**Abstract**: Applications that use Large Language Models (LLMs) are becoming widespread, making the identification of system vulnerabilities increasingly important. Automated Red Teaming accelerates this effort by using an LLM to generate and execute attacks against target systems. Attack generators are evaluated using the Attack Success Rate (ASR) the sample mean calculated over the judgment of success for each attack. In this paper, we introduce a method for optimizing attack generator prompts that applies ASR to individual attacks. By repeating each attack multiple times against a randomly seeded target, we measure an attack's discoverability the expectation of the individual attack success. This approach reveals exploitable patterns that inform prompt optimization, ultimately enabling more robust evaluation and refinement of generators.

摘要: 使用大型语言模型（LLM）的应用程序正在变得广泛，使得系统漏洞的识别变得越来越重要。自动红色团队通过使用LLM生成和执行针对目标系统的攻击来加速这一工作。使用攻击成功率（ASB）评估攻击生成器，即根据每次攻击的成功判断计算的样本平均值。本文中，我们介绍了一种优化攻击生成器提示的方法，该方法将ASB应用于单个攻击。通过对随机种子目标重复多次每次攻击，我们衡量攻击的可预见性和个人攻击成功的预期。这种方法揭示了可利用的模式，为及时优化提供信息，最终实现对生成器的更稳健的评估和细化。



## **10. Secure Tug-of-War (SecTOW): Iterative Defense-Attack Training with Reinforcement Learning for Multimodal Model Security**

安全拔河（SecTOW）：通过强化学习实现多模式模型安全的迭代防御攻击训练 cs.CR

10 pages, 4 figures

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.22037v1) [paper-pdf](http://arxiv.org/pdf/2507.22037v1)

**Authors**: Muzhi Dai, Shixuan Liu, Zhiyuan Zhao, Junyu Gao, Hao Sun, Xuelong Li

**Abstract**: The rapid advancement of multimodal large language models (MLLMs) has led to breakthroughs in various applications, yet their security remains a critical challenge. One pressing issue involves unsafe image-query pairs--jailbreak inputs specifically designed to bypass security constraints and elicit unintended responses from MLLMs. Compared to general multimodal data, such unsafe inputs are relatively sparse, which limits the diversity and richness of training samples available for developing robust defense models. Meanwhile, existing guardrail-type methods rely on external modules to enforce security constraints but fail to address intrinsic vulnerabilities within MLLMs. Traditional supervised fine-tuning (SFT), on the other hand, often over-refuses harmless inputs, compromising general performance. Given these challenges, we propose Secure Tug-of-War (SecTOW), an innovative iterative defense-attack training method to enhance the security of MLLMs. SecTOW consists of two modules: a defender and an auxiliary attacker, both trained iteratively using reinforcement learning (GRPO). During the iterative process, the attacker identifies security vulnerabilities in the defense model and expands jailbreak data. The expanded data are then used to train the defender, enabling it to address identified security vulnerabilities. We also design reward mechanisms used for GRPO to simplify the use of response labels, reducing dependence on complex generative labels and enabling the efficient use of synthetic data. Additionally, a quality monitoring mechanism is used to mitigate the defender's over-refusal of harmless inputs and ensure the diversity of the jailbreak data generated by the attacker. Experimental results on safety-specific and general benchmarks demonstrate that SecTOW significantly improves security while preserving general performance.

摘要: 多模式大型语言模型（MLLM）的快速发展导致了各种应用的突破，但其安全性仍然是一个严峻的挑战。一个紧迫的问题涉及不安全的图像查询对--专门设计用于绕过安全约束并引发MLLM的意外响应的越狱输入。与一般的多峰数据相比，此类不安全的输入相对稀疏，这限制了可用于开发稳健防御模型的训练样本的多样性和丰富性。与此同时，现有的护栏型方法依赖外部模块来强制执行安全约束，但无法解决MLLM内的固有漏洞。另一方面，传统的监督微调（SFT）经常过度拒绝无害的输入，从而损害总体性能。鉴于这些挑战，我们提出了安全拔河（SecTOW），这是一种创新的迭代防御攻击训练方法，旨在增强MLLM的安全性。SecTOW由两个模块组成：防御者和辅助攻击者，两者都使用强化学习（GRPO）迭代训练。在迭代过程中，攻击者识别防御模型中的安全漏洞并扩展越狱数据。然后使用扩展的数据来训练防御者，使其能够解决已识别的安全漏洞。我们还设计了用于GRPO的奖励机制，以简化响应标签的使用，减少对复杂生成标签的依赖，并实现合成数据的高效使用。此外，还使用质量监控机制来减轻防御者对无害输入的过度拒绝，并确保攻击者生成的越狱数据的多样性。针对特定安全性基准和通用基准的实验结果表明，SecTOW在保持一般性能的同时显着提高了安全性。



## **11. Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is**

任何人都可以越狱：针对LLM和T2 I的预算攻击 cs.CV

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21820v1) [paper-pdf](http://arxiv.org/pdf/2507.21820v1)

**Authors**: Ahmed B Mustafa, Zihan Ye, Yang Lu, Michael P Pound, Shreyank N Gowda

**Abstract**: Despite significant advancements in alignment and content moderation, large language models (LLMs) and text-to-image (T2I) systems remain vulnerable to prompt-based attacks known as jailbreaks. Unlike traditional adversarial examples requiring expert knowledge, many of today's jailbreaks are low-effort, high-impact crafted by everyday users with nothing more than cleverly worded prompts. This paper presents a systems-style investigation into how non-experts reliably circumvent safety mechanisms through techniques such as multi-turn narrative escalation, lexical camouflage, implication chaining, fictional impersonation, and subtle semantic edits. We propose a unified taxonomy of prompt-level jailbreak strategies spanning both text-output and T2I models, grounded in empirical case studies across popular APIs. Our analysis reveals that every stage of the moderation pipeline, from input filtering to output validation, can be bypassed with accessible strategies. We conclude by highlighting the urgent need for context-aware defenses that reflect the ease with which these jailbreaks can be reproduced in real-world settings.

摘要: 尽管在对齐和内容审核方面取得了重大进步，但大型语言模型（LLM）和文本到图像（T2 I）系统仍然容易受到基于预算的攻击（即越狱）。与需要专业知识的传统对抗示例不同，今天的许多越狱都是由日常用户精心设计的，只需措辞巧妙的提示即可。本文对非专家如何通过多回合叙事升级、词汇伪装、隐含链接、虚构模仿和微妙的语义编辑等技术可靠地规避安全机制进行了系统式的调查。我们基于流行API的实证案例研究，提出了跨越文本输出和T2 I模型的预算级越狱策略的统一分类。我们的分析表明，审核管道的每个阶段，从输入过滤到输出验证，都可以通过可访问的策略绕过。最后，我们强调了对上下文感知防御的迫切需要，以反映这些越狱可以在现实世界环境中轻松复制的情况。



## **12. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

隐性对抗培训提高了法学硕士对持续有害行为的稳健性 cs.LG

Code at https://github.com/aengusl/latent-adversarial-training.  Models at https://huggingface.co/LLM-LAT

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2407.15549v3) [paper-pdf](http://arxiv.org/pdf/2407.15549v3)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.

摘要: 大型语言模型（LLM）通常会以不受欢迎的方式运行，而这些方式明确进行了微调。例如，LLM红色团队文献产生了各种“越狱”技术，从经过微调的无害模型中引出有害文本。最近关于红色团队、模型编辑和可解释性的工作表明，这一挑战源于（对抗性）微调如何在很大程度上用于抑制而不是消除LLM中不受欢迎的功能。之前的工作引入了潜在对抗训练（LAT），作为提高对广泛类型失败的稳健性的一种方法。这些先前的作品考虑了无针对性的潜在空间攻击，其中对手扰乱潜在激活，以最大限度地增加理想行为示例的损失。无目标LAT可以提供通用类型的稳健性，但不会利用有关特定故障模式的信息。在这里，我们尝试了有针对性的LAT，其中对手试图最大限度地减少特定竞争任务的损失。我们发现它可以增强各种最先进的方法。首先，我们使用有针对性的LAT来提高对越狱的稳健性，以减少数量级的计算来超越强大的R2 D2基线。其次，我们使用它来在不知道触发器的情况下更有效地删除后门。最后，我们使用它以一种对重新学习更稳健的方式更有效地忘记特定不需要任务的知识。总体而言，我们的结果表明，有针对性的LAT可以成为防御LLM有害行为的有效工具。



## **13. PRISM: Programmatic Reasoning with Image Sequence Manipulation for LVLM Jailbreaking**

PRism：用于LVLM越狱的具有图像序列操作的程序推理 cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21540v1) [paper-pdf](http://arxiv.org/pdf/2507.21540v1)

**Authors**: Quanchen Zou, Zonghao Ying, Moyang Chen, Wenzhuo Xu, Yisong Xiao, Yakai Li, Deyue Zhang, Dongdong Yang, Zhao Liu, Xiangzheng Zhang

**Abstract**: The increasing sophistication of large vision-language models (LVLMs) has been accompanied by advances in safety alignment mechanisms designed to prevent harmful content generation. However, these defenses remain vulnerable to sophisticated adversarial attacks. Existing jailbreak methods typically rely on direct and semantically explicit prompts, overlooking subtle vulnerabilities in how LVLMs compose information over multiple reasoning steps. In this paper, we propose a novel and effective jailbreak framework inspired by Return-Oriented Programming (ROP) techniques from software security. Our approach decomposes a harmful instruction into a sequence of individually benign visual gadgets. A carefully engineered textual prompt directs the sequence of inputs, prompting the model to integrate the benign visual gadgets through its reasoning process to produce a coherent and harmful output. This makes the malicious intent emergent and difficult to detect from any single component. We validate our method through extensive experiments on established benchmarks including SafeBench and MM-SafetyBench, targeting popular LVLMs. Results show that our approach consistently and substantially outperforms existing baselines on state-of-the-art models, achieving near-perfect attack success rates (over 0.90 on SafeBench) and improving ASR by up to 0.39. Our findings reveal a critical and underexplored vulnerability that exploits the compositional reasoning abilities of LVLMs, highlighting the urgent need for defenses that secure the entire reasoning process.

摘要: 随着大型视觉语言模型（LVLM）的日益复杂，旨在防止有害内容生成的安全对齐机制也取得了进步。然而，这些防御系统仍然容易受到复杂的对抗攻击。现有的越狱方法通常依赖于直接且语义明确的提示，忽略了LVLM如何通过多个推理步骤组成信息的微妙漏洞。本文受到软件安全领域的面向返回编程（opp）技术的启发，提出了一种新颖且有效的越狱框架。我们的方法将有害的指令分解为一系列单独良性的视觉小工具。精心设计的文本提示引导输入序列，促使模型通过其推理过程集成良性视觉小工具，以产生连贯且有害的输出。这使得恶意意图变得紧急，并且难以从任何单个组件中检测到。我们通过对SafeBench和MM-SafetyBench等既定基准进行广泛实验来验证我们的方法，目标是流行的LVLM。结果表明，我们的方法始终且大幅优于最先进模型上的现有基线，实现了近乎完美的攻击成功率（SafeBench上超过0.90），并将ASB提高高达0.39。我们的研究结果揭示了一个关键且未充分探索的漏洞，该漏洞利用了LVLM的合成推理能力，凸显了对保护整个推理过程的防御措施的迫切需求。



## **14. Can We End the Cat-and-Mouse Game? Simulating Self-Evolving Phishing Attacks with LLMs and Genetic Algorithms**

我们能结束猫鼠游戏吗？使用LLM和遗传算法模拟自我进化的网络钓鱼攻击 cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21538v1) [paper-pdf](http://arxiv.org/pdf/2507.21538v1)

**Authors**: Seiji Sato, Tetsushi Ohki, Masakatsu Nishigaki

**Abstract**: Anticipating emerging attack methodologies is crucial for proactive cybersecurity. Recent advances in Large Language Models (LLMs) have enabled the automated generation of phishing messages and accelerated research into potential attack techniques. However, predicting future threats remains challenging due to reliance on existing training data. To address this limitation, we propose a novel framework that integrates LLM-based phishing attack simulations with a genetic algorithm in a psychological context, enabling phishing strategies to evolve dynamically through adversarial interactions with simulated victims. Through simulations using Llama 3.1, we demonstrate that (1) self-evolving phishing strategies employ increasingly sophisticated psychological manipulation techniques, surpassing naive LLM-generated attacks, (2) variations in a victim's prior knowledge significantly influence the evolution of attack strategies, and (3) adversarial interactions between evolving attacks and adaptive defenses create a cat-and-mouse dynamic, revealing an inherent asymmetry in cybersecurity -- attackers continuously refine their methods, whereas defenders struggle to comprehensively counter all evolving threats. Our approach provides a scalable, cost-effective method for analyzing the evolution of phishing strategies and defenses, offering insights into future social engineering threats and underscoring the necessity of proactive cybersecurity measures.

摘要: 预测新出现的攻击方法对于主动网络安全至关重要。大型语言模型（LLM）的最新进展使网络钓鱼消息的自动生成成为可能，并加速了对潜在攻击技术的研究。然而，由于依赖于现有的训练数据，预测未来的威胁仍然具有挑战性。为了解决这一限制，我们提出了一种新的框架，集成了基于LLM的网络钓鱼攻击模拟与遗传算法在心理背景下，使网络钓鱼策略通过与模拟受害者的对抗性交互动态演变。通过使用Llama 3.1的模拟，我们证明了（1）自我进化的网络钓鱼策略采用了越来越复杂的心理操纵技术，超越了天真的LLM生成的攻击，（2）受害者先验知识的变化显着影响攻击策略的演变，（3）不断进化的攻击和适应性防御之间的对抗相互作用创造了猫鼠动态，暴露了网络安全固有的不对称性--攻击者不断完善他们的方法，而防御者则难以全面应对所有不断变化的威胁。我们的方法提供了一种可扩展、具有成本效益的方法来分析网络钓鱼策略和防御的演变，提供了对未来社会工程威胁的见解，并强调了主动网络安全措施的必要性。



## **15. Memorization in Fine-Tuned Large Language Models**

微调大型语言模型中的精简化 cs.CL

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21009v1) [paper-pdf](http://arxiv.org/pdf/2507.21009v1)

**Authors**: Danil Savine, Muni Sreenivas Pydi, Jamal Atif, Olivier Cappé

**Abstract**: This study investigates the mechanisms and factors influencing memorization in fine-tuned large language models (LLMs), with a focus on the medical domain due to its privacy-sensitive nature. We examine how different aspects of the fine-tuning process affect a model's propensity to memorize training data, using the PHEE dataset of pharmacovigilance events.   Our research employs two main approaches: a membership inference attack to detect memorized data, and a generation task with prompted prefixes to assess verbatim reproduction. We analyze the impact of adapting different weight matrices in the transformer architecture, the relationship between perplexity and memorization, and the effect of increasing the rank in low-rank adaptation (LoRA) fine-tuning.   Key findings include: (1) Value and Output matrices contribute more significantly to memorization compared to Query and Key matrices; (2) Lower perplexity in the fine-tuned model correlates with increased memorization; (3) Higher LoRA ranks lead to increased memorization, but with diminishing returns at higher ranks.   These results provide insights into the trade-offs between model performance and privacy risks in fine-tuned LLMs. Our findings have implications for developing more effective and responsible strategies for adapting large language models while managing data privacy concerns.

摘要: 这项研究调查了影响微调大型语言模型（LLM）记忆的机制和因素，重点关注医学领域，因为其隐私敏感的性质。我们使用药物警戒事件的PHEE数据集，研究微调过程的不同方面如何影响模型记忆训练数据的倾向。   我们的研究采用了两种主要方法：检测记忆数据的隶属度推理攻击，以及具有提示性前置的生成任务，以评估逐字复制。我们分析了在Transformer架构中适应不同权重矩阵的影响、困惑度和记忆之间的关系，以及在低等级适应（LoRA）微调中增加等级的影响。   主要调查结果包括：（1）与查询和关键矩阵相比，价值和输出矩阵对记忆的贡献更大;（2）微调模型中较低的困惑度与记忆的增加相关;（3）较高的LoRA排名会导致记忆的增加，但收益递减。   这些结果为微调LLM中模型性能和隐私风险之间的权衡提供了见解。我们的研究结果对于制定更有效、更负责任的策略来适应大型语言模型，同时管理数据隐私问题具有影响。



## **16. A Large Language Model-Supported Threat Modeling Framework for Transportation Cyber-Physical Systems**

运输网络物理系统支持的大语言模型威胁建模框架 cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2506.00831v2) [paper-pdf](http://arxiv.org/pdf/2506.00831v2)

**Authors**: M Sabbir Salek, Mashrur Chowdhury, Muhaimin Bin Munir, Yuchen Cai, Mohammad Imtiaz Hasan, Jean-Michel Tine, Latifur Khan, Mizanur Rahman

**Abstract**: Existing threat modeling frameworks related to transportation cyber-physical systems (CPS) are often narrow in scope, labor-intensive, and require substantial cybersecurity expertise. To this end, we introduce the Transportation Cybersecurity and Resiliency Threat Modeling Framework (TraCR-TMF), a large language model (LLM)-based threat modeling framework for transportation CPS that requires limited cybersecurity expert intervention. TraCR-TMF identifies threats, potential attack techniques, and relevant countermeasures for transportation CPS. Three LLM-based approaches support these identifications: (i) a retrieval-augmented generation approach requiring no cybersecurity expert intervention, (ii) an in-context learning approach with low expert intervention, and (iii) a supervised fine-tuning approach with moderate expert intervention. TraCR-TMF offers LLM-based attack path identification for critical assets based on vulnerabilities across transportation CPS entities. Additionally, it incorporates the Common Vulnerability Scoring System (CVSS) scores of known exploited vulnerabilities to prioritize threat mitigations. The framework was evaluated through two cases. First, the framework identified relevant attack techniques for various transportation CPS applications, 73% of which were validated by cybersecurity experts as correct. Second, the framework was used to identify attack paths for a target asset in a real-world cyberattack incident. TraCR-TMF successfully predicted exploitations, like lateral movement of adversaries, data exfiltration, and data encryption for ransomware, as reported in the incident. These findings show the efficacy of TraCR-TMF in transportation CPS threat modeling, while reducing the need for extensive involvement of cybersecurity experts. To facilitate real-world adoptions, all our codes are shared via an open-source repository.

摘要: 与交通网络物理系统（CPS）相关的现有威胁建模框架通常范围狭窄、劳动密集型，并且需要大量的网络安全专业知识。为此，我们引入了交通网络安全和弹性威胁建模框架（TraCR-SYS），这是一个基于大型语言模型（LLM）的交通CPS威胁建模框架，需要有限的网络安全专家干预。TraCR-SYS识别交通CPS的威胁、潜在攻击技术和相关对策。三种基于LLM的方法支持这些识别：（i）不需要网络安全专家干预的检索增强生成方法，（ii）具有低专家干预的上下文学习方法，（iii）具有适度专家干预的监督微调方法。TraCR-SYS根据运输CPS实体之间的漏洞为关键资产提供基于LLM的攻击路径识别。此外，它还结合了已知被利用漏洞的通用漏洞评分系统（CVD）评分，以确定威胁缓解的优先顺序。该框架通过两个案例进行了评估。首先，该框架确定了各种交通CPS应用程序的相关攻击技术，其中73%经网络安全专家验证为正确。其次，该框架用于识别现实世界网络攻击事件中目标资产的攻击路径。正如该事件中所报道的那样，TraCR-SYS成功预测了攻击行为，例如对手的横向移动、数据泄露和勒索软件的数据加密。这些发现表明了TraCR-SYS在交通CPS威胁建模中的功效，同时减少了网络安全专家广泛参与的需要。为了促进现实世界的采用，我们所有的代码都通过开源存储库共享。



## **17. Enhancing Jailbreak Attacks on LLMs via Persona Prompts**

通过女神异闻录加强对LLM的越狱攻击 cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.22171v1) [paper-pdf](http://arxiv.org/pdf/2507.22171v1)

**Authors**: Zheng Zhang, Peilin Zhao, Deheng Ye, Hao Wang

**Abstract**: Jailbreak attacks aim to exploit large language models (LLMs) by inducing them to generate harmful content, thereby revealing their vulnerabilities. Understanding and addressing these attacks is crucial for advancing the field of LLM safety. Previous jailbreak approaches have mainly focused on direct manipulations of harmful intent, with limited attention to the impact of persona prompts. In this study, we systematically explore the efficacy of persona prompts in compromising LLM defenses. We propose a genetic algorithm-based method that automatically crafts persona prompts to bypass LLM's safety mechanisms. Our experiments reveal that: (1) our evolved persona prompts reduce refusal rates by 50-70% across multiple LLMs, and (2) these prompts demonstrate synergistic effects when combined with existing attack methods, increasing success rates by 10-20%. Our code and data are available at https://github.com/CjangCjengh/Generic_Persona.

摘要: 越狱攻击旨在通过诱导大型语言模型（LLM）生成有害内容来利用它们，从而暴露它们的漏洞。了解和解决这些攻击对于推进LLM安全领域至关重要。之前的越狱方法主要集中在对有害意图的直接操纵上，对角色提示的影响关注有限。在这项研究中，我们系统地探讨了角色提示在损害LLM防御方面的功效。我们提出了一种基于遗传算法的方法，可以自动制作角色提示以绕过LLM的安全机制。我们的实验表明：（1）我们进化的角色提示将多个LLM的拒绝率降低50-70%，并且（2）这些提示与现有的攻击方法结合时表现出协同效应，将成功率提高10- 20%。我们的代码和数据可在https://github.com/CjangCjengh/Generic_Persona上获取。



## **18. Uncovering Gradient Inversion Risks in Practical Language Model Training**

揭示实用语言模型训练中的梯度倒置风险 cs.LG

15 Pages, 5 figures, 10 tables. Accepted by ACM CCS 2024

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21198v1) [paper-pdf](http://arxiv.org/pdf/2507.21198v1)

**Authors**: Xinguo Feng, Zhongkui Ma, Zihan Wang, Eu Joe Chegne, Mengyao Ma, Alsharif Abuadbba, Guangdong Bai

**Abstract**: The gradient inversion attack has been demonstrated as a significant privacy threat to federated learning (FL), particularly in continuous domains such as vision models. In contrast, it is often considered less effective or highly dependent on impractical training settings when applied to language models, due to the challenges posed by the discrete nature of tokens in text data. As a result, its potential privacy threats remain largely underestimated, despite FL being an emerging training method for language models. In this work, we propose a domain-specific gradient inversion attack named Grab (gradient inversion with hybrid optimization). Grab features two alternating optimization processes to address the challenges caused by practical training settings, including a simultaneous optimization on dropout masks between layers for improved token recovery and a discrete optimization for effective token sequencing. Grab can recover a significant portion (up to 92.9% recovery rate) of the private training data, outperforming the attack strategy of utilizing discrete optimization with an auxiliary model by notable improvements of up to 28.9% recovery rate in benchmark settings and 48.5% recovery rate in practical settings. Grab provides a valuable step forward in understanding this privacy threat in the emerging FL training mode of language models.

摘要: 梯度倒置攻击已被证明是对联邦学习（FL）的重大隐私威胁，特别是在视觉模型等连续领域中。相比之下，由于文本数据中标记的离散性带来的挑战，当应用于语言模型时，它通常被认为效率较低或高度依赖于不切实际的训练设置。因此，尽管FL是一种新兴的语言模型训练方法，但其潜在的隐私威胁在很大程度上被低估了。在这项工作中，我们提出了一种名为Grab的特定领域梯度倒置攻击（具有混合优化的梯度倒置）。Grab具有两个交替的优化流程，以解决实际训练设置带来的挑战，包括对层之间的脱落掩蔽进行同时优化以改善令牌恢复，以及对有效的令牌排序进行离散优化。Grab可以恢复很大一部分（高达92.9%的恢复率）的私人训练数据，优于利用辅助模型的离散优化的攻击策略，基准设置中恢复率高达28.9%，实际设置中恢复率高达48.5%。Grab为理解新兴的FL语言模型训练模式中的这种隐私威胁向前迈出了宝贵的一步。



## **19. Accidental Vulnerability: Factors in Fine-Tuning that Shift Model Safeguards**

意外漏洞：微调中改变模型保障措施的因素 cs.CL

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2505.16789v2) [paper-pdf](http://arxiv.org/pdf/2505.16789v2)

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Abstract**: As large language models (LLMs) gain popularity, their vulnerability to adversarial attacks emerges as a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can inadvertently introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Vulnerability, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity across multiple experimental datasets. We then evaluate the adversarial robustness of these fine-tuned models, analyzing persona shifts and interpretability traits to understand how dataset factors contribute to attack success rates. Lastly, we explore causal relationships that offer new insights into adversarial defense strategies, highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_vulnerability.

摘要: 随着大型语言模型（LLM）的普及，它们对对抗攻击的脆弱性成为主要问题。虽然通常使用特定领域数据集的微调模型来提高模型性能，但它可能会无意中在基础模型中引入漏洞。在这项工作中，我们调查了意外漏洞，即由微调数据特征引起的意外漏洞。我们首先识别多个实验数据集中的潜在相关因素，例如语言特征、语义相似性和毒性。然后，我们评估这些微调模型的对抗稳健性，分析角色转变和可解释性特征，以了解数据集因素如何影响攻击成功率。最后，我们探索了因果关系，为对抗性防御策略提供了新的见解，强调了数据集设计在保留模型对齐方面的关键作用。我们的代码可在https://github.com/psyonp/accidental_vulnerability上获取。



## **20. Security Challenges in AI Agent Deployment: Insights from a Large Scale Public Competition**

人工智能代理部署中的安全挑战：大规模公开竞争的见解 cs.AI

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.20526v1) [paper-pdf](http://arxiv.org/pdf/2507.20526v1)

**Authors**: Andy Zou, Maxwell Lin, Eliot Jones, Micha Nowak, Mateusz Dziemian, Nick Winter, Alexander Grattan, Valent Nathanael, Ayla Croft, Xander Davies, Jai Patel, Robert Kirk, Nate Burnikell, Yarin Gal, Dan Hendrycks, J. Zico Kolter, Matt Fredrikson

**Abstract**: Recent advances have enabled LLM-powered AI agents to autonomously execute complex tasks by combining language model reasoning with tools, memory, and web access. But can these systems be trusted to follow deployment policies in realistic environments, especially under attack? To investigate, we ran the largest public red-teaming competition to date, targeting 22 frontier AI agents across 44 realistic deployment scenarios. Participants submitted 1.8 million prompt-injection attacks, with over 60,000 successfully eliciting policy violations such as unauthorized data access, illicit financial actions, and regulatory noncompliance. We use these results to build the Agent Red Teaming (ART) benchmark - a curated set of high-impact attacks - and evaluate it across 19 state-of-the-art models. Nearly all agents exhibit policy violations for most behaviors within 10-100 queries, with high attack transferability across models and tasks. Importantly, we find limited correlation between agent robustness and model size, capability, or inference-time compute, suggesting that additional defenses are needed against adversarial misuse. Our findings highlight critical and persistent vulnerabilities in today's AI agents. By releasing the ART benchmark and accompanying evaluation framework, we aim to support more rigorous security assessment and drive progress toward safer agent deployment.

摘要: 最近的进展使LLM驱动的AI代理能够通过将语言模型推理与工具，内存和Web访问相结合来自主执行复杂的任务。但是，这些系统在现实环境中，特别是在受到攻击的情况下，是否能够遵循部署策略呢？为了进行调查，我们进行了迄今为止最大规模的公开红队竞赛，目标是44个现实部署场景中的22个前沿人工智能代理。参与者提交了180万次注入攻击，其中超过60，000次成功引发了违反政策的行为，如未经授权的数据访问，非法金融行为和不遵守监管规定。我们使用这些结果来构建Agent Red Teaming（ART）基准（一组精心策划的高影响力攻击），并在19个最先进的模型上对其进行评估。几乎所有代理在10-100个查询内的大多数行为都会违反策略，并且攻击跨模型和任务的可转移性很高。重要的是，我们发现代理稳健性与模型大小、能力或推断时计算之间的相关性有限，这表明需要针对对抗性滥用采取额外的防御措施。我们的研究结果凸显了当今人工智能代理中的关键且持续存在的漏洞。通过发布ART基准和附带的评估框架，我们的目标是支持更严格的安全评估并推动更安全的代理部署取得进展。



## **21. More is Less: The Pitfalls of Multi-Model Synthetic Preference Data in DPO Safety Alignment**

多即少：DPO安全调整中多模型合成偏好数据的陷阱 cs.AI

This version includes updated results and expanded discussion

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2504.02193v3) [paper-pdf](http://arxiv.org/pdf/2504.02193v3)

**Authors**: Yifan Wang, Runjin Chen, Bolian Li, David Cho, Yihe Deng, Ruqi Zhang, Tianlong Chen, Zhangyang Wang, Ananth Grama, Junyuan Hong

**Abstract**: Aligning large language models (LLMs) with human values is an increasingly critical step in post-training. Direct Preference Optimization (DPO) has emerged as a simple, yet effective alternative to reinforcement learning from human feedback (RLHF). Synthetic preference data with its low cost and high quality enable effective alignment through single- or multi-model generated preference data. Our study reveals a striking, safety-specific phenomenon associated with DPO alignment: Although multi-model generated data enhances performance on general tasks (ARC, Hellaswag, MMLU, TruthfulQA, Winogrande) by providing diverse responses, it also tends to facilitate reward hacking during training. This can lead to a high attack success rate (ASR) when models encounter jailbreaking prompts. The issue is particularly pronounced when employing stronger models like GPT-4o or larger models in the same family to generate chosen responses paired with target model self-generated rejected responses, resulting in dramatically poorer safety outcomes. Furthermore, with respect to safety, using solely self-generated responses (single-model generation) for both chosen and rejected pairs significantly outperforms configurations that incorporate responses from stronger models, whether used directly as chosen data or as part of a multi-model response pool. We demonstrate that multi-model preference data exhibits high linear separability between chosen and rejected responses, which allows models to exploit superficial cues rather than internalizing robust safety constraints. Our experiments, conducted on models from the Llama, Mistral, and Qwen families, consistently validate these findings.

摘要: 将大型语言模型（LLM）与人类价值观相结合是后期训练中越来越关键的一步。直接偏好优化（DPO）已经成为一种简单而有效的替代人类反馈强化学习（RLHF）的方法。合成偏好数据以其低成本和高质量使得能够通过单个或多个模型生成的偏好数据进行有效对齐。我们的研究揭示了一个与DPO对齐相关的引人注目的安全特定现象：尽管多模型生成的数据通过提供不同的响应来增强一般任务（ARC，Hellaswag，MMLU，TruthfulQA，Winogrande）的性能，但它也倾向于促进训练期间的奖励黑客行为。当模型遇到越狱提示时，这可能会导致很高的攻击成功率（ASB）。当在同一系列中使用GPT-4 o等更强大的模型或更大的模型来生成与目标模型自生成的拒绝响应配对的选择响应时，这个问题尤其明显，导致安全性结果显着较差。此外，就安全性而言，对选择和拒绝的对仅使用自生成响应（单模型生成）显着优于包含来自更强模型的响应的配置，无论是直接用作选择的数据还是作为多模型响应池的一部分。我们证明，多模型偏好数据在选择的和拒绝的响应之间表现出高度的线性可分性，这使得模型能够利用表面线索，而不是内化强大的安全约束。我们对Lama、Mistral和Qwen家族的模型进行的实验一致验证了这些发现。



## **22. Interpretable Anomaly-Based DDoS Detection in AI-RAN with XAI and LLMs**

具有XAI和LLM的AI-RAN中基于可解释异常的DDOS检测 cs.CR

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2507.21193v1) [paper-pdf](http://arxiv.org/pdf/2507.21193v1)

**Authors**: Sotiris Chatzimiltis, Mohammad Shojafar, Mahdi Boloursaz Mashhadi, Rahim Tafazolli

**Abstract**: Next generation Radio Access Networks (RANs) introduce programmability, intelligence, and near real-time control through intelligent controllers, enabling enhanced security within the RAN and across broader 5G/6G infrastructures. This paper presents a comprehensive survey highlighting opportunities, challenges, and research gaps for Large Language Models (LLMs)-assisted explainable (XAI) intrusion detection (IDS) for secure future RAN environments. Motivated by this, we propose an LLM interpretable anomaly-based detection system for distributed denial-of-service (DDoS) attacks using multivariate time series key performance measures (KPMs), extracted from E2 nodes, within the Near Real-Time RAN Intelligent Controller (Near-RT RIC). An LSTM-based model is trained to identify malicious User Equipment (UE) behavior based on these KPMs. To enhance transparency, we apply post-hoc local explainability methods such as LIME and SHAP to interpret individual predictions. Furthermore, LLMs are employed to convert technical explanations into natural-language insights accessible to non-expert users. Experimental results on real 5G network KPMs demonstrate that our framework achieves high detection accuracy (F1-score > 0.96) while delivering actionable and interpretable outputs.

摘要: 下一代无线电接入网络（RAN）通过智能控制器引入可编程性、智能性和近实时控制，增强RAN内以及更广泛的5G/6 G基础设施的安全性。本文提出了一项全面的调查，重点介绍了用于安全未来RAN环境的大型语言模型（LLM）辅助可解释（XAI）入侵检测（IDS）的机遇、挑战和研究差距。出于此动机，我们提出了一种基于LLM可解释异常的检测系统，用于分布式拒绝服务（DDOS）攻击，使用从近实时RAN智能控制器（Near-RT RIC）内的E2节点提取的多元时间序列关键性能指标（KPMs）。训练基于LSTM的模型以基于这些KPI识别恶意用户设备（UE）行为。为了提高透明度，我们应用LIME和SHAP等事后局部解释方法来解释单个预测。此外，LLM用于将技术解释转化为非专家用户可以访问的自然语言见解。在真实5G网络上的实验结果表明，我们的框架实现了高检测准确性（F1-score > 0.96），同时提供可操作和可解释的输出。



## **23. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

通过跨模式提示注射操纵多模式代理 cs.CV

16 pages, 5 figures

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2504.14348v4) [paper-pdf](http://arxiv.org/pdf/2504.14348v4)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this paper, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach incorporates two key coordinated components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to construct the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms state-of-the-art attacks, achieving at least a +30.1% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.

摘要: 多模式大型语言模型的出现通过将语言和视觉模式与外部数据源集成来重新定义了代理范式，使代理能够更好地解释人类指令并执行日益复杂的任务。然而，在本文中，我们发现了多模式代理中一个以前被忽视的关键安全漏洞：跨模式提示注入攻击。为了利用这个漏洞，我们提出了CrossInib，这是一种新型攻击框架，其中攻击者在多种模式中嵌入对抗性扰动，以与目标恶意内容保持一致，允许外部指令劫持代理的决策过程并执行未经授权的任务。我们的方法包含两个关键的协调组成部分。首先，我们引入了视觉潜在对齐，基于文本到图像生成模型，优化视觉嵌入空间中恶意指令的对抗特征，确保对抗图像巧妙地编码恶意任务执行的线索。随后，我们提出了文本指导增强，其中利用大型语言模型通过对抗性Meta提示来构建黑匣子防御系统提示，并生成恶意文本命令，该命令引导代理的输出更好地遵守攻击者的请求。大量实验表明，我们的方法优于最先进的攻击，在不同任务中的攻击成功率至少提高+30.1%。此外，我们还验证了攻击在现实世界的多模式自治代理中的有效性，强调了其对安全关键应用程序的潜在影响。



## **24. SDD: Self-Degraded Defense against Malicious Fine-tuning**

SDD：针对恶意微调的自我降级防御 cs.CR

Accepted by ACL2025

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2507.21182v1) [paper-pdf](http://arxiv.org/pdf/2507.21182v1)

**Authors**: Zixuan Chen, Weikai Lu, Xin Lin, Ziqian Zeng

**Abstract**: Open-source Large Language Models (LLMs) often employ safety alignment methods to resist harmful instructions. However, recent research shows that maliciously fine-tuning these LLMs on harmful data can easily bypass these safeguards. To counter this, we theoretically uncover why malicious fine-tuning succeeds and identify potential defense strategies. Building on the theoretical analysis, we introduce the Self-Degraded Defense (SDD) framework. SDD encourages LLMs to produce high-quality but irrelevant responses to harmful prompts. When attackers attempt malicious fine-tuning, the general capability of the LLM aligned by SDD will significantly decrease, rendering it incapable of following harmful instructions. Our experimental results confirm SDD's effectiveness against such attacks.

摘要: 开源大型语言模型（LLM）通常采用安全对齐方法来抵抗有害指令。然而，最近的研究表明，针对有害数据恶意微调这些LLM可以轻松绕过这些保障措施。为了解决这个问题，我们从理论上揭示了恶意微调成功的原因并确定了潜在的防御策略。在理论分析的基础上，我们介绍了自降级防御（SDD）框架。SDD鼓励LLM对有害提示做出高质量但不相关的回应。当攻击者尝试恶意微调时，通过SDD对齐的LLM的一般能力将显着下降，使其无法遵循有害指令。我们的实验结果证实了SDD对抗此类攻击的有效性。



## **25. Risks & Benefits of LLMs & GenAI for Platform Integrity, Healthcare Diagnostics, Financial Trust and Compliance, Cybersecurity, Privacy & AI Safety: A Comprehensive Survey, Roadmap & Implementation Blueprint**

LLM和GenAI在平台完整性、医疗保健诊断、财务信任和合规、网络安全、隐私和人工智能安全方面的风险和收益：全面调查、路线图和实施蓝图 cs.CR

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2506.12088v2) [paper-pdf](http://arxiv.org/pdf/2506.12088v2)

**Authors**: Kiarash Ahi

**Abstract**: Large Language Models (LLMs) and generative AI (GenAI) systems, such as ChatGPT, Claude, Gemini, LLaMA, and Copilot (by OpenAI, Anthropic, Google, Meta, and Microsoft, respectively), are reshaping digital platforms and app ecosystems while introducing critical challenges in cybersecurity, privacy, and platform integrity. Our analysis reveals alarming trends: LLM-assisted malware is projected to rise from 2% (2021) to 50% (2025); AI-generated Google reviews grew nearly tenfold (1.2% in 2021 to 12.21% in 2023, expected to reach 30% by 2025); AI scam reports surged 456%; misinformation sites increased over 1500%; and deepfake attacks are projected to rise over 900% in 2025. In finance, LLM-driven threats like synthetic identity fraud and AI-generated scams are accelerating. Platforms such as JPMorgan Chase, Stripe, and Plaid deploy LLMs for fraud detection, regulation parsing, and KYC/AML automation, reducing fraud loss by up to 21% and accelerating onboarding by 40-60%. LLM-facilitated code development has driven mobile app submissions from 1.8 million (2020) to 3.0 million (2024), projected to reach 3.6 million (2025). To address AI threats, platforms like Google Play, Apple App Store, GitHub Copilot, TikTok, Facebook, and Amazon deploy LLM-based defenses, highlighting their dual nature as both threat sources and mitigation tools. In clinical diagnostics, LLMs raise concerns about accuracy, bias, and safety, necessitating strong governance. Drawing on 445 references, this paper surveys LLM/GenAI and proposes a strategic roadmap and operational blueprint integrating policy auditing (such as CCPA and GDPR compliance), fraud detection, and demonstrates an advanced LLM-DA stack with modular components, multi-LLM routing, agentic memory, and governance layers. We provide actionable insights, best practices, and real-world case studies for scalable trust and responsible innovation.

摘要: 大型语言模型（LLM）和生成式人工智能（GenAI）系统，例如ChatGPT、Claude、Gemini、LLaMA和Copilot（分别由OpenAI、Anthropic、Google、Meta和Microsoft开发），正在重塑数字平台和应用程序生态系统，同时在网络安全、隐私和平台完整性方面引入了关键挑战。我们的分析揭示了令人震惊的趋势：LLM辅助的恶意软件预计将从2%（2021年）上升到50%（2025年）;人工智能生成的谷歌评论增长近十倍（2021年为1.2%至2023年为12.21%，预计到2025年将达到30%）;人工智能诈骗报告激增456%;错误信息网站增长超过1500%;预计2025年Deepfake攻击将增加900%以上。在金融领域，合成身份欺诈和人工智能生成的诈骗等LLM驱动的威胁正在加速发展。摩根大通、Stripe和Plaid等平台部署LLM进行欺诈检测、监管解析和KWC/ML自动化，将欺诈损失减少高达21%，并将入职速度加快40 - 60%。LLM推动的代码开发已将移动应用程序提交量从180万（2020年）增加到300万（2024年），预计将达到360万（2025年）。为了应对人工智能威胁，Google Play、Apple App Store、GitHub Copilot、TikTok、Facebook和亚马逊等平台部署了基于LLM的防御，凸显了它们既是威胁源又是缓解工具的双重性质。在临床诊断中，LLM提出了对准确性、偏差和安全性的担忧，需要强有力的治理。本文借鉴445篇参考文献，调查了LLM/GenAI，并提出了集成政策审计（例如CCPA和GDPR合规性）、欺诈检测的战略路线图和运营蓝图，并演示了具有模块化组件、多LLM路由、代理内存和治理层的先进LLM-DA堆栈。我们提供可操作的见解、最佳实践和现实世界的案例研究，以实现可扩展的信任和负责任的创新。



## **26. LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning**

LoX：低等级外推损害了LLM的安全性，防止微调 cs.LG

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2506.15606v3) [paper-pdf](http://arxiv.org/pdf/2506.15606v3)

**Authors**: Gabriel J. Perin, Runjin Chen, Xuxi Chen, Nina S. T. Hirata, Zhangyang Wang, Junyuan Hong

**Abstract**: Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX.

摘要: 大型语言模型（LLM）在现实世界的应用程序中已变得不可或缺。然而，它们的广泛采用引发了严重的安全问题，特别是在应对社会有害问题时。尽管为通过对齐来提高模型安全性做出了巨大努力，但对齐的模型的安全保护仍然可能会因随后的微调而受到破坏--即使额外的训练数据看起来是良性的。在本文中，我们通过经验证明，这种漏洞源于LLM参数中安全关键的低阶子空间对微调的敏感性。基于这一见解，我们提出了一种新型的免训练方法，称为低等级外推（LoX），通过外推对齐LLM的安全子空间来增强安全鲁棒性。我们的实验结果证实了LoX的有效性，证明了针对良性和恶意微调攻击的鲁棒性得到了显着提高，同时保留了模型对新任务的适应性。例如，LoX导致面临良性或恶意微调攻击的攻击成功率（ASB）绝对降低11%至54%。通过研究参数的ASB格局，我们将LoX的成功归因于外推将LLM参数移动到更平坦的区域，从而对扰动不太敏感。该代码可在github.com/VITA-Group/LoX上获取。



## **27. MOCHA: Are Code Language Models Robust Against Multi-Turn Malicious Coding Prompts?**

MOCHA：代码语言模型对多轮恶意编码预测是否稳健？ cs.CL

Winner Defender Team at Amazon Nova AI Challenge 2025

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2507.19598v1) [paper-pdf](http://arxiv.org/pdf/2507.19598v1)

**Authors**: Muntasir Wahed, Xiaona Zhou, Kiet A. Nguyen, Tianjiao Yu, Nirav Diwan, Gang Wang, Dilek Hakkani-Tür, Ismini Lourentzou

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly enhanced their code generation capabilities. However, their robustness against adversarial misuse, particularly through multi-turn malicious coding prompts, remains underexplored. In this work, we introduce code decomposition attacks, where a malicious coding task is broken down into a series of seemingly benign subtasks across multiple conversational turns to evade safety filters. To facilitate systematic evaluation, we introduce \benchmarkname{}, a large-scale benchmark designed to evaluate the robustness of code LLMs against both single-turn and multi-turn malicious prompts. Empirical results across open- and closed-source models reveal persistent vulnerabilities, especially under multi-turn scenarios. Fine-tuning on MOCHA improves rejection rates while preserving coding ability, and importantly, enhances robustness on external adversarial datasets with up to 32.4% increase in rejection rates without any additional supervision.

摘要: 大型语言模型（LLM）的最新进展显着增强了它们的代码生成能力。然而，它们对对抗性滥用（特别是通过多轮恶意编码提示）的鲁棒性仍然没有得到充分的研究。在这项工作中，我们引入了代码分解攻击，其中恶意编码任务被分解为跨多个对话回合的一系列看似良性的子任务，以逃避安全过滤器。为了促进系统性评估，我们引入了\benchmarkName{}，这是一个大规模基准测试，旨在评估代码LLM针对单轮和多轮恶意提示的稳健性。开源和闭源模型的实证结果揭示了持续存在的漏洞，特别是在多回合场景下。对MOCHA的微调提高了拒绝率，同时保留了编码能力，重要的是，增强了外部对抗数据集的鲁棒性，在没有任何额外监督的情况下，拒绝率提高了32.4%。



## **28. Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation**

越狱的大型语言扩散模型：揭示基于扩散的文本生成中隐藏的安全缺陷 cs.CL

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2507.19227v1) [paper-pdf](http://arxiv.org/pdf/2507.19227v1)

**Authors**: Yuanhe Zhang, Fangzhou Xie, Zhenhong Zhou, Zherui Li, Hao Chen, Kun Wang, Yufei Guo

**Abstract**: Large Language Diffusion Models (LLDMs) exhibit comparable performance to LLMs while offering distinct advantages in inference speed and mathematical reasoning tasks.The precise and rapid generation capabilities of LLDMs amplify concerns of harmful generations, while existing jailbreak methodologies designed for Large Language Models (LLMs) prove limited effectiveness against LLDMs and fail to expose safety vulnerabilities.Successful defense cannot definitively resolve harmful generation concerns, as it remains unclear whether LLDMs possess safety robustness or existing attacks are incompatible with diffusion-based architectures.To address this, we first reveal the vulnerability of LLDMs to jailbreak and demonstrate that attack failure in LLDMs stems from fundamental architectural differences.We present a PArallel Decoding jailbreak (PAD) for diffusion-based language models. PAD introduces Multi-Point Attention Attack, which guides parallel generative processes toward harmful outputs that inspired by affirmative response patterns in LLMs. Experimental evaluations across four LLDMs demonstrate that PAD achieves jailbreak attack success rates by 97%, revealing significant safety vulnerabilities. Furthermore, compared to autoregressive LLMs of the same size, LLDMs increase the harmful generation speed by 2x, significantly highlighting risks of uncontrolled misuse.Through comprehensive analysis, we provide an investigation into LLDM architecture, offering critical insights for the secure deployment of diffusion-based language models.

摘要: 大型语言扩散模型（LLDM）表现出与LLM相当的性能，同时在推理速度和数学推理任务方面提供明显优势。LLDM的精确和快速生成能力加剧了对有害世代的担忧，而现有的越狱方法是为大型语言模型（LLM）设计的证明针对LLDM的有效性有限，并且未能暴露安全漏洞。成功的防御无法明确解决有害发电问题，由于目前尚不清楚LLDM是否具有安全鲁棒性，或者现有的攻击是否与基于扩散的架构不兼容。为了解决这个问题，我们首先揭示了LLDM的越狱脆弱性，并证明LLDM中的攻击失败源于基本的架构差异。我们提出了一种基于扩散的语言模型的PArSYS解码越狱（PAD）。PAD引入了多点注意力攻击，它引导并行生成过程转向有害输出，其灵感来自LLM中的肯定反应模式。对四个LLDM的实验评估表明，PAD的越狱攻击成功率高达97%，揭示了显着的安全漏洞。此外，与相同规模的自回归LLM相比，LLDM将有害生成速度提高了2倍，显着凸显了不受控制的滥用风险。通过全面分析，我们对LLDM架构进行了调查，为基于扩散的语言模型的安全部署提供了重要见解。



## **29. Blind Spot Navigation: Evolutionary Discovery of Sensitive Semantic Concepts for LVLMs**

盲点导航：LVLM敏感语义概念的进化发现 cs.CV

The paper needs major revisions, so it is being withdrawn

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2505.15265v2) [paper-pdf](http://arxiv.org/pdf/2505.15265v2)

**Authors**: Zihao Pan, Yu Tong, Weibin Wu, Jingyi Wang, Lifeng Chen, Zhe Zhao, Jiajia Wei, Yitong Qiao, Zibin Zheng

**Abstract**: Adversarial attacks aim to generate malicious inputs that mislead deep models, but beyond causing model failure, they cannot provide certain interpretable information such as ``\textit{What content in inputs make models more likely to fail?}'' However, this information is crucial for researchers to specifically improve model robustness. Recent research suggests that models may be particularly sensitive to certain semantics in visual inputs (such as ``wet,'' ``foggy''), making them prone to errors. Inspired by this, in this paper we conducted the first exploration on large vision-language models (LVLMs) and found that LVLMs indeed are susceptible to hallucinations and various errors when facing specific semantic concepts in images. To efficiently search for these sensitive concepts, we integrated large language models (LLMs) and text-to-image (T2I) models to propose a novel semantic evolution framework. Randomly initialized semantic concepts undergo LLM-based crossover and mutation operations to form image descriptions, which are then converted by T2I models into visual inputs for LVLMs. The task-specific performance of LVLMs on each input is quantified as fitness scores for the involved semantics and serves as reward signals to further guide LLMs in exploring concepts that induce LVLMs. Extensive experiments on seven mainstream LVLMs and two multimodal tasks demonstrate the effectiveness of our method. Additionally, we provide interesting findings about the sensitive semantics of LVLMs, aiming to inspire further in-depth research.

摘要: 对抗性攻击旨在生成误导深度模型的恶意输入，但除了导致模型失败之外，它们无法提供某些可解释的信息，例如'\textit{输入中的哪些内容使模型更有可能失败？}”“然而，这些信息对于研究人员专门提高模型稳健性至关重要。最近的研究表明，模型可能对视觉输入中的某些语义（例如“湿”、“雾”）特别敏感，这使得它们容易出错。受此启发，本文对大型视觉语言模型（LVLM）进行了首次探索，发现LVLM在面对图像中的特定语义概念时确实容易产生幻觉和各种错误。为了有效地搜索这些敏感概念，我们集成了大型语言模型（LLM）和文本到图像（T2 I）模型，提出了一种新颖的语义进化框架。随机初始化的语义概念经过基于LLM的交叉和变异操作以形成图像描述，然后由T2 I模型将其转换为LVLM的视觉输入。LVLM在每个输入上的特定任务性能被量化为所涉及语义的适应度分数，并作为奖励信号，以进一步指导LLM探索引发LVLM的概念。对七种主流LVLM和两种多模式任务的广泛实验证明了我们方法的有效性。此外，我们还提供了有关LVLM敏感语义的有趣发现，旨在激发进一步的深入研究。



## **30. KGV: Integrating Large Language Models with Knowledge Graphs for Cyber Threat Intelligence Credibility Assessment**

KGV：将大型语言模型与知识图集成，用于网络威胁情报可信度评估 cs.CR

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2408.08088v2) [paper-pdf](http://arxiv.org/pdf/2408.08088v2)

**Authors**: Zongzong Wu, Fengxiao Tang, Ming Zhao, Yufeng Li

**Abstract**: Cyber threat intelligence (CTI) is a crucial tool to prevent sophisticated, organized, and weaponized cyber attacks. However, few studies have focused on the credibility assessment of CTI, and this work still requires manual analysis by cybersecurity experts. In this paper, we propose Knowledge Graph-based Verifier (KGV), the first framework integrating large language models (LLMs) with simple structured knowledge graphs (KGs) for automated CTI credibility assessment. Unlike entity-centric KGs, KGV constructs paragraph-level semantic graphs where nodes represent text segments connected through similarity analysis, which effectively enhances the semantic understanding ability of the model, reduces KG density and greatly improves response speed. Experimental results demonstrate that our KGV outperforms state-of-the-art fact reasoning methods on the CTI-200 dataset, achieving a 5.7\% improvement in F1. Additionally, it shows strong scalability on factual QA and fake news detection datasets. Compared to entity-based knowledge graphs (KGs) for equivalent-length texts, our structurally simple KG reduces node quantities by nearly two-thirds while boosting precision by 1.7\% and cutting response time by 46.7\%. In addition, we have created and publicly released the first CTI credibility assessment dataset, CTI-200. Distinct from CTI identification datasets, CTI-200 refines CTI summaries and key sentences to focus specifically on credibility assessment.

摘要: 网络威胁情报（RTI）是防止复杂、有组织和武器化网络攻击的重要工具。然而，很少有研究关注RTI的可信度评估，这项工作仍然需要网络安全专家的手动分析。在本文中，我们提出了基于知识图的验证器（KGV），这是第一个将大型语言模型（LLM）与简单结构化知识图（KG）集成的框架，用于自动化RTI可信度评估。与以实体为中心的KG不同，KGV构建段落级语义图，其中节点代表通过相似度分析连接的文本片段，有效增强了模型的语义理解能力，降低了KG密度，大大提高了响应速度。实验结果表明，我们的KGV在RTI-200数据集上的表现优于最先进的事实推理方法，在F1中实现了5.7%的改进。此外，它在事实QA和假新闻检测数据集上表现出强大的可扩展性。与等长文本的基于实体的知识图（KG）相比，我们结构简单的KG将节点数量减少了近三分之二，同时将精确度提高了1.7%，并将响应时间缩短了46.7%。此外，我们还创建并公开发布了第一个RTI可信度评估数据集RTI-200。与RTI识别数据集不同，RTI-200改进了RTI摘要和关键句子，专门关注可信度评估。



## **31. Kill two birds with one stone: generalized and robust AI-generated text detection via dynamic perturbations**

一箭双雕：通过动态扰动进行广义且稳健的AI生成文本检测 cs.CL

Accepted by NAACL 2025 main conference

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2504.21019v2) [paper-pdf](http://arxiv.org/pdf/2504.21019v2)

**Authors**: Yinghan Zhou, Juan Wen, Wanli Peng, Yiming Xue, Ziwei Zhang, Zhengxian Wu

**Abstract**: The growing popularity of large language models has raised concerns regarding the potential to misuse AI-generated text (AIGT). It becomes increasingly critical to establish an excellent AIGT detection method with high generalization and robustness. However, existing methods either focus on model generalization or concentrate on robustness. The unified mechanism, to simultaneously address the challenges of generalization and robustness, is less explored. In this paper, we argue that robustness can be view as a specific form of domain shift, and empirically reveal an intrinsic mechanism for model generalization of AIGT detection task. Then, we proposed a novel AIGT detection method (DP-Net) via dynamic perturbations introduced by a reinforcement learning with elaborated reward and action. Experimentally, extensive results show that the proposed DP-Net significantly outperforms some state-of-the-art AIGT detection methods for generalization capacity in three cross-domain scenarios. Meanwhile, the DP-Net achieves best robustness under two text adversarial attacks. The code is publicly available at https://github.com/CAU-ISS-Lab/AIGT-Detection-Evade-Detection/tree/main/DP-Net.

摘要: 大型语言模型的日益流行引发了人们对滥用人工智能生成文本（AIGT）可能性的担忧。建立一种具有高度通用性和鲁棒性的优秀AIGT检测方法变得越来越重要。然而，现有方法要么专注于模型推广，要么专注于鲁棒性。同时解决通用性和稳健性挑战的统一机制很少被探索。本文认为鲁棒性可以被视为域转移的一种特定形式，并从经验上揭示了AIGT检测任务模型推广的内在机制。然后，我们提出了一种新型的AIGT检测方法（DP-Net），该方法通过强化学习引入的动态扰动，并具有详细的奖励和动作。实验结果表明，在三种跨域场景中，拟议的DP-Net的概括能力明显优于一些最先进的AIGT检测方法。与此同时，DP-Net在两次文本对抗攻击下实现了最佳的鲁棒性。该代码可在https://github.com/CAU-ISS-Lab/AIGT-Detection-Evade-Detection/tree/main/DP-Net上公开获取。



## **32. Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities**

模型篡改攻击能够更严格地评估LLM能力 cs.CR

Accepted to TMLR

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2502.05209v4) [paper-pdf](http://arxiv.org/pdf/2502.05209v4)

**Authors**: Zora Che, Stephen Casper, Robert Kirk, Anirudh Satheesh, Stewart Slocum, Lev E McKinney, Rohit Gandikota, Aidan Ewart, Domenic Rosati, Zichu Wu, Zikui Cai, Bilal Chughtai, Yarin Gal, Furong Huang, Dylan Hadfield-Menell

**Abstract**: Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot fully evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together, these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.

摘要: 对大型语言模型（LLM）风险和能力的评估越来越多地被纳入人工智能风险管理和治理框架中。目前，大多数风险评估都是通过设计从系统中引发有害行为的输入来进行的。然而，这种方法有两个局限性。首先，投入产出评估无法完全评估开权模型的现实风险。其次，在任何特定的投入-产出评估期间识别的行为只能下限模型的最坏可能情况的投入-产出行为。作为引发有害行为的补充方法，我们建议使用模型篡改攻击来评估LLM，该攻击允许修改潜在激活或权重。我们使用最先进的技术来消除有害的LLM功能，以对抗一系列5个输入空间和6个模型篡改攻击。除了对这些方法进行比较之外，我们还表明：（1）模型对能力启发攻击的弹性取决于低维鲁棒性子空间;（2）模型篡改攻击的成功率可以根据经验预测并为保持的输入空间攻击的成功提供保守估计;（3）最先进的取消学习方法可以在16个微调步骤内轻松取消。总之，这些结果突出了抑制有害LLM能力的困难，并表明模型篡改攻击比单独的输入空间攻击能够进行更严格的评估。



## **33. Noise Contrastive Estimation-based Matching Framework for Low-Resource Security Attack Pattern Recognition**

基于噪音对比估计的低资源安全攻击模式识别匹配框架 cs.LG

accepted at EACL 2024, in ARR October 2023

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2401.10337v4) [paper-pdf](http://arxiv.org/pdf/2401.10337v4)

**Authors**: Tu Nguyen, Nedim Šrndić, Alexander Neth

**Abstract**: Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning process of the matching model despite constrained resources.

摘要: 战术、技术和程序（TTP）代表网络安全领域中复杂的攻击模式，在文本知识库中进行了局部描述。在网络安全写作中识别TTP（通常称为TTP映射）是一项重要且具有挑战性的任务。传统的学习方法通常针对经典多类或多标签分类设置中的问题。由于类数量较多，这种设置阻碍了模型的学习能力（即，TTP）、标签分布不可避免的倾斜性以及标签空间复杂的分层结构。我们在不同的学习范式中制定了问题，其中将文本分配给TTP标签是由两者之间的直接语义相似性决定的，从而降低了仅在大标签空间上竞争的复杂性。为此，我们提出了一种具有有效的基于采样的学习比较机制的神经匹配架构，在资源有限的情况下促进匹配模型的学习过程。



## **34. A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1**

基于LLM的论点分类的全面研究：从LLAMA到GPT-4 o到Deepseek-R1 cs.CL

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.08621v2) [paper-pdf](http://arxiv.org/pdf/2507.08621v2)

**Authors**: Marcin Pietroń, Rafał Olszowski, Jakub Gomułka, Filip Gampel, Andrzej Tomski

**Abstract**: Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.

摘要: 论据挖掘（AM）是一个跨学科研究领域，集成了逻辑、哲学、语言学、修辞学、法学、心理学和计算机科学的见解。它涉及自动识别和提取论点成分（例如前提和主张），以及检测它们之间的关系（例如支持、攻击或中立）。最近，该领域取得了显着的进步，特别是随着大型语言模型（LLM）的出现，与传统方法和其他深度学习模型相比，LLM提高了分析和提取参数语义的效率。有许多用于测试和验证LLM质量的基准，但在公开可用的论点分类数据库中仍然缺乏有关这些模型操作的研究和结果。本文使用Args.me和UKP等不同数据集对LLM进行了一项研究。测试的模型包括GPT、Llama和DeepSeek的版本，以及包含思想链算法的推理增强变体。结果表明，ChatGPT-4 o在论点分类基准方面优于其他。在包含推理能力的模型中，Deepseek-R1显示出其优越性。然而，尽管具有优势，GPT-4 o和Deepseek-R1仍然会犯错误。讨论了所有模型最常见的错误。据我们所知，所介绍的工作是首次使用LLM和提示算法对上述数据集进行更广泛的分析。该工作还揭示了已知提示算法在论据分析中的一些弱点，同时指出了改进的方向。这项工作的附加值是对可用论点数据集的深入分析并展示其缺点。



## **35. BadReasoner: Planting Tunable Overthinking Backdoors into Large Reasoning Models for Fun or Profit**

BadReasoner：在大型推理模型中植入可调节的过度思考后门以获取乐趣或利润 cs.CL

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18305v1) [paper-pdf](http://arxiv.org/pdf/2507.18305v1)

**Authors**: Biao Yi, Zekun Fei, Jianing Geng, Tong Li, Lihai Nie, Zheli Liu, Yiming Li

**Abstract**: Large reasoning models (LRMs) have emerged as a significant advancement in artificial intelligence, representing a specialized class of large language models (LLMs) designed to tackle complex reasoning tasks. The defining characteristic of LRMs lies in their extensive chain-of-thought (CoT) reasoning capabilities. In this paper, we identify a previously unexplored attack vector against LRMs, which we term "overthinking backdoors". We advance this concept by proposing a novel tunable backdoor, which moves beyond simple on/off attacks to one where an attacker can precisely control the extent of the model's reasoning verbosity. Our attack is implemented through a novel data poisoning methodology. It pairs a tunable trigger-where the number of repetitions signals the desired intensity-with a correspondingly verbose CoT response. These responses are programmatically generated by instructing a teacher LLM to inject a controlled number of redundant refinement steps into a correct reasoning process. The approach preserves output correctness, which ensures stealth and establishes the attack as a pure resource-consumption vector. Extensive empirical results on various LRMs demonstrate that our method can reliably trigger a controllable, multi-fold increase in the length of the reasoning process, without degrading the final answer's correctness. Our source code is available at https://github.com/FZaKK/BadReasoner.

摘要: 大型推理模型（LRM）已成为人工智能领域的一项重大进步，代表了一类专门用于处理复杂推理任务的大型语言模型（LRM）。LRM的定义特征在于其广泛的思想链（CoT）推理能力。在本文中，我们识别了一种以前未探索过的针对LRM的攻击载体，我们将其称为“过度思考后门”。我们通过提出一种新颖的可调后门来推进这一概念，它超越了简单的开/关攻击，转向攻击者可以精确控制模型推理冗长程度的攻击。我们的攻击是通过一种新颖的数据中毒方法实施的。它将可调的发送器（重复次数表示所需强度）与相应详细的CoT响应配对。这些响应是通过指示教师LLM将受控数量的冗余细化步骤注入到正确的推理过程中来编程生成的。该方法保留了输出的正确性，从而确保了隐蔽性并将攻击建立为纯粹的资源消耗载体。对各种LRM的大量经验结果表明，我们的方法可以可靠地触发推理过程长度的可控、多倍增加，而不会降低最终答案的正确性。我们的源代码可在https://github.com/FZaKK/BadReasoner上获取。



## **36. Auto-SGCR: Automated Generation of Smart Grid Cyber Range Using IEC 61850 Standard Models**

Auto-SGCR：使用IEC 61850标准模型自动生成智能电网网络范围 cs.CR

12 pages

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18249v1) [paper-pdf](http://arxiv.org/pdf/2507.18249v1)

**Authors**: Muhammad M. Roomi, S. M. Suhail Hussain, Ee-Chien Chang, David M. Nicol, Daisuke Mashima

**Abstract**: Digitalization of power grids have made them increasingly susceptible to cyber-attacks in the past decade. Iterative cybersecurity testing is indispensable to counter emerging attack vectors and to ensure dependability of critical infrastructure. Furthermore, these can be used to evaluate cybersecurity configuration, effectiveness of the cybersecurity measures against various attack vectors, as well as to train smart grid cybersecurity experts defending the system. Enabling extensive experiments narrows the gap between academic research and production environment. A high-fidelity cyber range is vital as it is often infeasible to conduct such experiments and training using production environment. However, the design and implementation of cyber range requires extensive domain knowledge of physical and cyber aspect of the infrastructure. Furthermore, costs incurred for setup and maintenance of cyber range are significant. Moreover, most existing smart grid cyber ranges are designed as a one-off, proprietary system, and are limited in terms of configurability, accessibility, portability, and reproducibility. To address these challenges, an automated Smart grid Cyber Range generation framework is presented in this paper. Initially a human-/machine-friendly, XML-based modeling language called Smart Grid Modeling Language was defined, which incorporates IEC 61850 System Configuration Language files. Subsequently, a toolchain to parse SG-ML model files and automatically instantiate a functional smart grid cyber range was developed. The developed SG-ML models can be easily shared and/or modified to reproduce or customize for any cyber range. The application of Auto-SGCR is demonstrated through case studies with large-scale substation models. The toolchain along with example SG-ML models have been open-sourced.

摘要: 过去十年，电网的数字化使其越来越容易受到网络攻击。迭代网络安全测试对于对抗新出现的攻击载体和确保关键基础设施的可靠性是必不可少的。此外，这些可用于评估网络安全配置、网络安全措施针对各种攻击载体的有效性，以及培训保护系统的智能电网网络安全专家。实现广泛的实验缩小了学术研究和生产环境之间的差距。高保真度的网络范围至关重要，因为使用生产环境进行此类实验和培训通常是不可行的。然而，网络范围的设计和实施需要基础设施的物理和网络方面的广泛领域知识。此外，网络范围的设置和维护成本也很高。此外，大多数现有的智能电网网络范围都被设计为一次性的专有系统，并且在可配置性、可访问性、便携性和可重复性方面受到限制。为了应对这些挑战，本文提出了一种自动化智能电网Cyber Range生成框架。最初定义了一种人机友好、基于ML的建模语言，称为智能电网建模语言，它包含了IEC 61850系统配置语言文件。随后，开发了一个工具链来解析SG-ML模型文件并自动实例化功能智能电网网络范围。开发的SG-ML模型可以轻松共享和/或修改，以针对任何网络范围进行复制或定制。通过大型变电站模型的案例研究来演示Auto-SGCR的应用。工具链以及示例SG-ML模型已开源。



## **37. Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection**

使用GMTA保护RAG管道：一种基于对象的掩蔽令牌概率方法，用于有毒文档检测 cs.CL

18 pages, accepted to ACL Findings 2025

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18202v1) [paper-pdf](http://arxiv.org/pdf/2507.18202v1)

**Authors**: San Kim, Jonghwi Kim, Yejin Jeon, Gary Geunbae Lee

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing external knowledge for accurate and up-to-date responses. However, this reliance on external sources exposes a security risk, attackers can inject poisoned documents into the knowledge base to steer the generation process toward harmful or misleading outputs. In this paper, we propose Gradient-based Masked Token Probability (GMTP), a novel defense method to detect and filter out adversarially crafted documents. Specifically, GMTP identifies high-impact tokens by examining gradients of the retriever's similarity function. These key tokens are then masked, and their probabilities are checked via a Masked Language Model (MLM). Since injected tokens typically exhibit markedly low masked-token probabilities, this enables GMTP to easily detect malicious documents and achieve high-precision filtering. Experiments demonstrate that GMTP is able to eliminate over 90% of poisoned content while retaining relevant documents, thus maintaining robust retrieval and generation performance across diverse datasets and adversarial settings.

摘要: 检索增强生成（RAG）通过提供外部知识以提供准确和最新的响应来增强大型语言模型（LLM）。然而，这种对外部来源的依赖暴露了安全风险，攻击者可以将有毒文档注入知识库，以引导生成过程转向有害或误导性的输出。在本文中，我们提出了基于对象的掩蔽令牌概率（GMTA），这是一种新型防御方法，用于检测和过滤敌对精心设计的文档。具体来说，GMTA通过检查检索器相似性函数的梯度来识别高影响力代币。然后，这些关键令牌被屏蔽，并通过屏蔽语言模型（MLM）检查它们的概率。由于注入的令牌通常表现出明显较低的被屏蔽令牌概率，这使GMTA能够轻松检测恶意文档并实现高精度过滤。实验表明，GMat能够消除超过90%的有毒内容，同时保留相关文档，从而在不同数据集和对抗性设置中保持稳健的检索和生成性能。



## **38. Policy Disruption in Reinforcement Learning:Adversarial Attack with Large Language Models and Critical State Identification**

强化学习中的政策破坏：大型语言模型和关键状态识别的对抗性攻击 cs.LG

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18113v1) [paper-pdf](http://arxiv.org/pdf/2507.18113v1)

**Authors**: Junyong Jiang, Buwei Tian, Chenxing Xu, Songze Li, Lu Dong

**Abstract**: Reinforcement learning (RL) has achieved remarkable success in fields like robotics and autonomous driving, but adversarial attacks designed to mislead RL systems remain challenging. Existing approaches often rely on modifying the environment or policy, limiting their practicality. This paper proposes an adversarial attack method in which existing agents in the environment guide the target policy to output suboptimal actions without altering the environment. We propose a reward iteration optimization framework that leverages large language models (LLMs) to generate adversarial rewards explicitly tailored to the vulnerabilities of the target agent, thereby enhancing the effectiveness of inducing the target agent toward suboptimal decision-making. Additionally, a critical state identification algorithm is designed to pinpoint the target agent's most vulnerable states, where suboptimal behavior from the victim leads to significant degradation in overall performance. Experimental results in diverse environments demonstrate the superiority of our method over existing approaches.

摘要: 强化学习（RL）在机器人和自动驾驶等领域取得了显着的成功，但旨在误导强化学习系统的对抗性攻击仍然具有挑战性。现有的方法通常依赖于修改环境或政策，限制了其实用性。本文提出了一种对抗攻击方法，其中环境中现有的代理引导目标策略在不改变环境的情况下输出次优动作。我们提出了一个奖励迭代优化框架，该框架利用大型语言模型（LLM）来生成明确针对目标代理的脆弱性定制的对抗奖励，从而提高诱导目标代理进行次优决策的有效性。此外，关键状态识别算法旨在确定目标代理最脆弱的状态，其中受害者的次优行为导致总体性能显着下降。不同环境中的实验结果证明了我们的方法优于现有方法。



## **39. RECALLED: An Unbounded Resource Consumption Attack on Large Vision-Language Models**

重新命名：对大型视觉语言模型的无限资源消耗攻击 cs.CR

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18053v1) [paper-pdf](http://arxiv.org/pdf/2507.18053v1)

**Authors**: Haoran Gao, Yuanhe Zhang, Zhenhong Zhou, Lei Jiang, Fanyu Meng, Yujia Xiao, Kun Wang, Yang Liu, Junlan Feng

**Abstract**: Resource Consumption Attacks (RCAs) have emerged as a significant threat to the deployment of Large Language Models (LLMs). With the integration of vision modalities, additional attack vectors exacerbate the risk of RCAs in large vision-language models (LVLMs). However, existing red-teaming studies have largely overlooked visual inputs as a potential attack surface, resulting in insufficient mitigation strategies against RCAs in LVLMs. To address this gap, we propose RECALLED (\textbf{RE}source \textbf{C}onsumption \textbf{A}ttack on \textbf{L}arge Vision-\textbf{L}anguag\textbf{E} Mo\textbf{D}els), the first approach for exploiting visual modalities to trigger unbounded RCAs red-teaming. First, we present \textit{Vision Guided Optimization}, a fine-grained pixel-level optimization, to obtain \textit{Output Recall} adversarial perturbations, which can induce repeating output. Then, we inject the perturbations into visual inputs, triggering unbounded generations to achieve the goal of RCAs. Additionally, we introduce \textit{Multi-Objective Parallel Losses} to generate universal attack templates and resolve optimization conflicts when intending to implement parallel attacks. Empirical results demonstrate that RECALLED increases service response latency by over 26 $\uparrow$, resulting in an additional 20\% increase in GPU utilization and memory consumption. Our study exposes security vulnerabilities in LVLMs and establishes a red-teaming framework that can facilitate future defense development against RCAs.

摘要: 资源消耗攻击（RCA）已成为大型语言模型（LLM）部署的重大威胁。随着视觉模式的集成，额外的攻击载体加剧了大型视觉语言模型（LVLM）中RCA的风险。然而，现有的红色团队研究在很大程度上忽视了视觉输入作为潜在攻击表面，导致针对LVLM中RCA的缓解策略不足。为了解决这一差距，我们提出了RECALLED（\textBF{RE}source \textBF{C}onsumption \textBF{A}ttack on \textBF {L}arge Vision-\textBF {L}anguag\textBF{E} Mo\textBF{D}els），这是第一种利用视觉模式触发无界RCA红色分组的方法。首先，我们提出了\textit{Vision Guided Optimism}，一种细粒度像素级优化，以获得\textit{Exit Recall}对抗性扰动，这可能会导致重复输出。然后，我们将扰动注入视觉输入中，触发无限世代来实现RCA的目标。此外，我们还引入了\texttit {Multi-Observer并行损失}来生成通用攻击模板并在打算实施并行攻击时解决优化冲突。经验结果表明，RECALLED将服务响应延迟增加了超过26 $\uparrow$，导致图形处理器利用率和内存消耗额外增加20%。我们的研究揭示了LVLM中的安全漏洞，并建立了一个红色团队框架，可以促进未来针对RCA的防御开发。



## **40. ViGText: Deepfake Image Detection with Vision-Language Model Explanations and Graph Neural Networks**

ViGtext：利用视觉语言模型解析和图神经网络进行Deepfake图像检测 cs.CV

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18031v1) [paper-pdf](http://arxiv.org/pdf/2507.18031v1)

**Authors**: Ahmad ALBarqawi, Mahmoud Nazzal, Issa Khalil, Abdallah Khreishah, NhatHai Phan

**Abstract**: The rapid rise of deepfake technology, which produces realistic but fraudulent digital content, threatens the authenticity of media. Traditional deepfake detection approaches often struggle with sophisticated, customized deepfakes, especially in terms of generalization and robustness against malicious attacks. This paper introduces ViGText, a novel approach that integrates images with Vision Large Language Model (VLLM) Text explanations within a Graph-based framework to improve deepfake detection. The novelty of ViGText lies in its integration of detailed explanations with visual data, as it provides a more context-aware analysis than captions, which often lack specificity and fail to reveal subtle inconsistencies. ViGText systematically divides images into patches, constructs image and text graphs, and integrates them for analysis using Graph Neural Networks (GNNs) to identify deepfakes. Through the use of multi-level feature extraction across spatial and frequency domains, ViGText captures details that enhance its robustness and accuracy to detect sophisticated deepfakes. Extensive experiments demonstrate that ViGText significantly enhances generalization and achieves a notable performance boost when it detects user-customized deepfakes. Specifically, average F1 scores rise from 72.45% to 98.32% under generalization evaluation, and reflects the model's superior ability to generalize to unseen, fine-tuned variations of stable diffusion models. As for robustness, ViGText achieves an increase of 11.1% in recall compared to other deepfake detection approaches. When facing targeted attacks that exploit its graph-based architecture, ViGText limits classification performance degradation to less than 4%. ViGText uses detailed visual and textual analysis to set a new standard for detecting deepfakes, helping ensure media authenticity and information integrity.

摘要: Deepfake技术的迅速兴起会产生真实但具有欺诈性的数字内容，威胁着媒体的真实性。传统的Deepfake检测方法常常难以应对复杂的、定制的Deepfake，特别是在针对恶意攻击的概括性和鲁棒性方面。本文介绍了ViGtext，这是一种新颖的方法，它将图像与视觉大语言模型（VLLM）文本解释集成在基于图形的框架内，以改进深度伪造检测。ViGtext的新颖之处在于它将详细解释与视觉数据集成，因为它提供了比标题更具上下文感知性的分析，而标题通常缺乏具体性并且无法揭示微妙的不一致之处。ViGtext系统地将图像分为补丁，构建图像和文本图，并使用图形神经网络（GNN）将它们集成起来进行分析以识别深度造假。通过使用跨空间和频率域的多层特征提取，ViGtext捕获细节，增强其检测复杂深度造假的鲁棒性和准确性。大量实验表明，ViGtext显着增强了概括性，并在检测用户定制的深度造假时实现了显着的性能提升。具体来说，在概括性评估下，F1平均得分从72.45%上升到98.32%，反映了该模型推广到稳定扩散模型的不可见、微调变体的卓越能力。至于稳健性，与其他deepfake检测方法相比，ViGtext的召回率提高了11.1%。当面临利用其基于图形的架构的有针对性的攻击时，ViGtext将分类性能下降限制在4%以下。ViGtext使用详细的视觉和文本分析来设定检测深度造假的新标准，帮助确保媒体真实性和信息完整性。



## **41. Evaluating the Performance of AI Text Detectors, Few-Shot and Chain-of-Thought Prompting Using DeepSeek Generated Text**

使用DeepSeek生成的文本评估AI文本检测器、Few-shot和思想链打印的性能 cs.CL

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17944v1) [paper-pdf](http://arxiv.org/pdf/2507.17944v1)

**Authors**: Hulayyil Alshammari, Praveen Rao

**Abstract**: Large language models (LLMs) have rapidly transformed the creation of written materials. LLMs have led to questions about writing integrity, thereby driving the creation of artificial intelligence (AI) detection technologies. Adversarial attacks, such as standard and humanized paraphrasing, inhibit detectors' ability to detect machine-generated text. Previous studies have mainly focused on ChatGPT and other well-known LLMs and have shown varying accuracy across detectors. However, there is a clear gap in the literature about DeepSeek, a recently published LLM. Therefore, in this work, we investigate whether six generally accessible AI detection tools -- AI Text Classifier, Content Detector AI, Copyleaks, QuillBot, GPT-2, and GPTZero -- can consistently recognize text generated by DeepSeek. The detectors were exposed to the aforementioned adversarial attacks. We also considered DeepSeek as a detector by performing few-shot prompting and chain-of-thought reasoning (CoT) for classifying AI and human-written text. We collected 49 human-authored question-answer pairs from before the LLM era and generated matching responses using DeepSeek-v3, producing 49 AI-generated samples. Then, we applied adversarial techniques such as paraphrasing and humanizing to add 196 more samples. These were used to challenge detector robustness and assess accuracy impact. While QuillBot and Copyleaks showed near-perfect performance on original and paraphrased DeepSeek text, others -- particularly AI Text Classifier and GPT-2 -- showed inconsistent results. The most effective attack was humanization, reducing accuracy to 71% for Copyleaks, 58% for QuillBot, and 52% for GPTZero. Few-shot and CoT prompting showed high accuracy, with the best five-shot result misclassifying only one of 49 samples (AI recall 96%, human recall 100%).

摘要: 大型语言模型（LLM）迅速改变了书面材料的创建。LLM引发了有关写作完整性的质疑，从而推动了人工智能（AI）检测技术的创建。对抗性攻击，例如标准和人性化的重述，会抑制检测器检测机器生成文本的能力。之前的研究主要集中在ChatGPT和其他知名的LLM上，并表明不同探测器的准确性各不相同。然而，关于DeepSeek（最近出版的法学硕士）的文献中存在明显的空白。因此，在这项工作中，我们调查了六种通用的人工智能检测工具-- AI文本分类器、内容检测器AI、Copyleaks、QuillBot、GPT-2和GPTZero --是否能够一致地识别DeepSeek生成的文本。探测器遭受了上述对抗性攻击。我们还将DeepSeek视为一个检测器，通过执行少量提示和思维链推理（CoT）来对人工智能和人类书写的文本进行分类。我们收集了LLM时代之前的49个人类创作的问答对，并使用DeepSeek-v3生成匹配响应，生成了49个人工智能生成的样本。然后，我们应用了解释和人性化等对抗技术，添加了196个样本。这些用于挑战检测器的稳健性并评估准确性影响。虽然QuillBot和Copyleaks在原始和转述的DeepSeek文本上表现出近乎完美的性能，但其他文本--尤其是AI文本分类器和GPT-2 --却表现出不一致的结果。最有效的攻击是人性化，将Copyleaks的准确性降低到71%，QuillBot的准确性降低到58%，GPTZero的准确性降低到52%。Fe-shot和CoT提示显示出很高的准确性，最好的五次shot结果仅错误分类了49个样本中的一个（人工智能召回率96%，人类召回率100%）。



## **42. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的从弱到强越狱 cs.CL

ICML 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2401.17256v5) [paper-pdf](http://arxiv.org/pdf/2401.17256v5)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 大型语言模型（LLM）很容易受到越狱攻击，从而导致有害、不道德或有偏见的文本生成。然而，现有的越狱方法计算成本很高。本文中，我们提出了弱到强越狱攻击，这是一种针对对齐LLM的有效推理时间攻击，以产生有害文本。我们的关键直觉是基于这样的观察：越狱和对齐的模型仅在其初始解码分布上有所不同。从弱到强攻击的关键技术见解是使用两个较小的模型（一个安全的模型和一个不安全的模型）来对抗性地修改明显更大的安全模型的解码概率。我们评估了对来自3个组织的5个不同开源LLM的弱到强攻击。结果表明，我们的方法可以将两个数据集的未对准率提高到99%以上，每个示例只需向前传递一次。我们的研究揭示了在调整LLM时需要解决的紧迫安全问题。作为初步尝试，我们提出了一种防御策略来抵御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上获取



## **43. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

MEF：一个用于评估黑箱大语言模型脆弱性的能力感知多重加密框架 cs.CL

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2505.23404v4) [paper-pdf](http://arxiv.org/pdf/2505.23404v4)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have exposed critical vulnerabilities in Large Language Models (LLMs), enabling the circumvention of alignment safeguards through increasingly sophisticated prompt manipulations. Based on our experiments, we found that the effectiveness of jailbreak strategies is influenced by the comprehension ability of the attacked LLM. Building on this insight, we propose a capability-aware Multi-Encryption Framework (MEF) for evaluating vulnerabilities in black-box LLMs. Specifically, MEF first categorizes the comprehension ability level of the LLM, then applies different strategies accordingly: For models with limited comprehension ability, MEF adopts the Fu+En1 strategy, which integrates layered semantic mutations with an encryption technique, more effectively contributing to evasion of the LLM's defenses at the input and inference stages. For models with strong comprehension ability, MEF uses a more complex Fu+En1+En2 strategy, in which additional dual-ended encryption techniques are applied to the LLM's responses, further contributing to evasion of the LLM's defenses at the output stage. Experimental results demonstrate the effectiveness of our approach, achieving attack success rates of 98.9% on GPT-4o (29 May 2025 release) and 99.8% on GPT-4.1 (8 July 2025 release). Our work contributes to a deeper understanding of the vulnerabilities in current LLM alignment mechanisms.

摘要: 对抗性越狱攻击的最新进展暴露了大型语言模型（LLM）中的关键漏洞，从而能够通过日益复杂的提示操作来规避对齐保障措施。根据我们的实验，我们发现越狱策略的有效性受到被攻击LLM理解能力的影响。基于这一见解，我们提出了一个功能感知的多加密框架（MEF），用于评估黑匣子LLM中的漏洞。具体来说，MEF首先对LLM的理解能力水平进行分类，然后相应地应用不同的策略：对于理解能力有限的模型，MEF采用Du + En 1策略，将分层语义突变与加密技术集成在一起，更有效地帮助规避LLM在输入和推理阶段的防御。对于理解能力强的模型，MEF使用更复杂的Fu+ En 1 + En 2策略，其中额外的双端加密技术应用于LLM的响应，进一步有助于在输出阶段逃避LLM的防御。实验结果证明了我们方法的有效性，在GPT-4 o（2025年5月29日发布）和GPT-4.1（2025年7月8日发布）上的攻击成功率分别为98.9%和99.8%。我们的工作有助于更深入地了解当前LLM对齐机制中的漏洞。



## **44. Tab-MIA: A Benchmark Dataset for Membership Inference Attacks on Tabular Data in LLMs**

Tab-MIA：LLM中表格数据的成员推断攻击的基准数据集 cs.CR

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17259v1) [paper-pdf](http://arxiv.org/pdf/2507.17259v1)

**Authors**: Eyal German, Sagiv Antebi, Daniel Samira, Asaf Shabtai, Yuval Elovici

**Abstract**: Large language models (LLMs) are increasingly trained on tabular data, which, unlike unstructured text, often contains personally identifiable information (PII) in a highly structured and explicit format. As a result, privacy risks arise, since sensitive records can be inadvertently retained by the model and exposed through data extraction or membership inference attacks (MIAs). While existing MIA methods primarily target textual content, their efficacy and threat implications may differ when applied to structured data, due to its limited content, diverse data types, unique value distributions, and column-level semantics. In this paper, we present Tab-MIA, a benchmark dataset for evaluating MIAs on tabular data in LLMs and demonstrate how it can be used. Tab-MIA comprises five data collections, each represented in six different encoding formats. Using our Tab-MIA benchmark, we conduct the first evaluation of state-of-the-art MIA methods on LLMs finetuned with tabular data across multiple encoding formats. In the evaluation, we analyze the memorization behavior of pretrained LLMs on structured data derived from Wikipedia tables. Our findings show that LLMs memorize tabular data in ways that vary across encoding formats, making them susceptible to extraction via MIAs. Even when fine-tuned for as few as three epochs, models exhibit high vulnerability, with AUROC scores approaching 90% in most cases. Tab-MIA enables systematic evaluation of these risks and provides a foundation for developing privacy-preserving methods for tabular data in LLMs.

摘要: 大型语言模型（LLM）越来越多地针对表格数据进行训练，与非结构化文本不同，表格数据通常包含高度结构化和显式格式的个人可识别信息（PRI）。因此，隐私风险就会出现，因为敏感记录可能会被模型无意中保留，并通过数据提取或成员资格推断攻击（MIA）暴露。虽然现有的MIA方法主要针对文本内容，但由于其内容有限、数据类型多样、值分布独特和列级语义，当应用于结构化数据时，它们的功效和威胁含义可能会有所不同。在本文中，我们介绍了Tab-MIA，这是一个用于评估LLM中表格数据MIA的基准数据集，并演示了如何使用它。Tab-MIA由五个数据集合组成，每个集合以六种不同的编码格式表示。使用我们的Tab-MIA基准测试，我们对LLM进行了最先进的MIA方法的首次评估，这些LLM使用跨多种编码格式的表格数据进行了微调。在评估中，我们分析了预训练的LLM对源自维基百科表的结构化数据的记忆行为。我们的研究结果表明，LLM以不同编码格式的方式记忆表格数据，使其容易通过MIA提取。即使只微调了三个时期，模型也表现出很高的脆弱性，在大多数情况下，AUROC分数接近90%。Tab-MIA可以系统地评估这些风险，并为LLM中表格数据的隐私保护方法提供基础。



## **45. LLM4MEA: Data-free Model Extraction Attacks on Sequential Recommenders via Large Language Models**

LLM4EMA：通过大型语言模型对顺序推荐器进行无数据模型提取攻击 cs.IR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16969v1) [paper-pdf](http://arxiv.org/pdf/2507.16969v1)

**Authors**: Shilong Zhao, Fei Sun, Kaike Zhang, Shaoling Jing, Du Su, Zhichao Shi, Zhiyi Yin, Huawei Shen, Xueqi Cheng

**Abstract**: Recent studies have demonstrated the vulnerability of sequential recommender systems to Model Extraction Attacks (MEAs). MEAs collect responses from recommender systems to replicate their functionality, enabling unauthorized deployments and posing critical privacy and security risks. Black-box attacks in prior MEAs are ineffective at exposing recommender system vulnerabilities due to random sampling in data selection, which leads to misaligned synthetic and real-world distributions. To overcome this limitation, we propose LLM4MEA, a novel model extraction method that leverages Large Language Models (LLMs) as human-like rankers to generate data. It generates data through interactions between the LLM ranker and target recommender system. In each interaction, the LLM ranker analyzes historical interactions to understand user behavior, and selects items from recommendations with consistent preferences to extend the interaction history, which serves as training data for MEA. Extensive experiments demonstrate that LLM4MEA significantly outperforms existing approaches in data quality and attack performance, reducing the divergence between synthetic and real-world data by up to 64.98% and improving MEA performance by 44.82% on average. From a defensive perspective, we propose a simple yet effective defense strategy and identify key hyperparameters of recommender systems that can mitigate the risk of MEAs.

摘要: 最近的研究表明，顺序推荐系统的脆弱性模型提取攻击（MEA）。MEA收集来自推荐系统的响应以复制其功能，从而实现未经授权的部署并带来严重的隐私和安全风险。由于数据选择中的随机采样，先前MEA中的黑盒攻击在暴露推荐系统漏洞方面是无效的，这导致合成分布和真实世界分布不一致。为了克服这一限制，我们提出了LLM4MEA，一种新的模型提取方法，利用大型语言模型（LLM）作为类人的排名来生成数据。它通过LLM排名器和目标推荐系统之间的交互生成数据。在每次交互中，LLM排名器都会分析历史交互以了解用户行为，并从推荐中选择具有一致偏好的项目以扩展交互历史记录，该历史记录用作多边环境管理局的训练数据。大量实验表明，LLM 4 EMA在数据质量和攻击性能方面显着优于现有方法，将合成数据与现实数据之间的差异减少了高达64.98%，并平均提高了44.82%。从防御的角度来看，我们提出了一种简单而有效的防御策略，并识别了可以减轻多边环境协定风险的推荐系统的关键超参数。



## **46. When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs**

当LLM复制到思考时：揭示推理LLM中的复制引导攻击 cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16773v1) [paper-pdf](http://arxiv.org/pdf/2507.16773v1)

**Authors**: Yue Li, Xiao Li, Hao Wu, Yue Zhang, Fengyuan Xu, Xiuzhen Cheng, Sheng Zhong

**Abstract**: Large Language Models (LLMs) have become integral to automated code analysis, enabling tasks such as vulnerability detection and code comprehension. However, their integration introduces novel attack surfaces. In this paper, we identify and investigate a new class of prompt-based attacks, termed Copy-Guided Attacks (CGA), which exploit the inherent copying tendencies of reasoning-capable LLMs. By injecting carefully crafted triggers into external code snippets, adversaries can induce the model to replicate malicious content during inference. This behavior enables two classes of vulnerabilities: inference length manipulation, where the model generates abnormally short or excessively long reasoning traces; and inference result manipulation, where the model produces misleading or incorrect conclusions. We formalize CGA as an optimization problem and propose a gradient-based approach to synthesize effective triggers. Empirical evaluation on state-of-the-art reasoning LLMs shows that CGA reliably induces infinite loops, premature termination, false refusals, and semantic distortions in code analysis tasks. While highly effective in targeted settings, we observe challenges in generalizing CGA across diverse prompts due to computational constraints, posing an open question for future research. Our findings expose a critical yet underexplored vulnerability in LLM-powered development pipelines and call for urgent advances in prompt-level defense mechanisms.

摘要: 大型语言模型（LLM）已成为自动代码分析的组成部分，可以实现漏洞检测和代码理解等任务。然而，它们的集成引入了新颖的攻击表面。在本文中，我们识别并研究了一类新的基于预算的攻击，称为复制引导攻击（CGA），它利用了具有推理能力的LLM固有的复制倾向。通过将精心制作的触发器注入外部代码片段中，对手可以诱导模型在推理期间复制恶意内容。这种行为导致两类漏洞：推理长度操纵，其中模型生成异常短或过长的推理痕迹;和推理结果操纵，其中模型生成误导性或不正确的结论。我们将CGA形式化为一个优化问题，并提出一种基于梯度的方法来合成有效的触发器。对最先进推理LLM的实证评估表明，CGA可靠地在代码分析任务中引发无限循环、提前终止、错误拒绝和语义扭曲。虽然在目标环境中非常有效，但由于计算限制，我们观察到在不同提示中推广CGA存在挑战，这为未来的研究提出了一个悬而未决的问题。我们的研究结果揭示了LLM驱动的开发管道中一个关键但未充分探索的漏洞，并呼吁在预算级防御机制方面紧急取得进展。



## **47. From Text to Actionable Intelligence: Automating STIX Entity and Relationship Extraction**

从文本到可操作智能：自动化STIX实体和关系提取 cs.CR

This paper is accepted at RAID 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16576v1) [paper-pdf](http://arxiv.org/pdf/2507.16576v1)

**Authors**: Ahmed Lekssays, Husrev Taha Sencar, Ting Yu

**Abstract**: Sharing methods of attack and their effectiveness is a cornerstone of building robust defensive systems. Threat analysis reports, produced by various individuals and organizations, play a critical role in supporting security operations and combating emerging threats. To enhance the timeliness and automation of threat intelligence sharing, several standards have been established, with the Structured Threat Information Expression (STIX) framework emerging as one of the most widely adopted. However, generating STIX-compatible data from unstructured security text remains a largely manual, expert-driven process. To address this challenge, we introduce AZERG, a tool designed to assist security analysts in automatically generating structured STIX representations. To achieve this, we adapt general-purpose large language models for the specific task of extracting STIX-formatted threat data. To manage the complexity, the task is divided into four subtasks: entity detection (T1), entity type identification (T2), related pair detection (T3), and relationship type identification (T4). We apply task-specific fine-tuning to accurately extract relevant entities and infer their relationships in accordance with the STIX specification. To address the lack of training data, we compiled a comprehensive dataset with 4,011 entities and 2,075 relationships extracted from 141 full threat analysis reports, all annotated in alignment with the STIX standard. Our models achieved F1-scores of 84.43% for T1, 88.49% for T2, 95.47% for T3, and 84.60% for T4 in real-world scenarios. We validated their performance against a range of open- and closed-parameter models, as well as state-of-the-art methods, demonstrating improvements of 2-25% across tasks.

摘要: 共享攻击方法及其有效性是构建强大防御系统的基石。由各种个人和组织制作的威胁分析报告在支持安全行动和应对新出现的威胁方面发挥着至关重要的作用。为了提高威胁情报共享的及时性和自动化，已经制定了多项标准，其中结构化威胁信息表达（STIX）框架成为最广泛采用的标准之一。然而，从非结构化安全文本生成STIX兼容的数据在很大程度上仍然是一个手动、专家驱动的过程。为了应对这一挑战，我们引入了AZERG，这是一种旨在协助安全分析师自动生成结构化STIX表示的工具。为了实现这一目标，我们调整了通用大型语言模型来执行提取STIX格式威胁数据的特定任务。为了管理复杂性，该任务被分为四个子任务：实体检测（T1）、实体类型识别（T2）、相关对检测（T3）和关系类型识别（T4）。我们应用特定于任务的微调来准确提取相关实体并根据STIX规范推断它们的关系。为了解决缺乏训练数据的问题，我们编制了一个全面的数据集，其中包含从141份完整的威胁分析报告中提取的4，011个实体和2，075个关系，所有这些都按照STIX标准进行了注释。在现实世界场景中，我们的模型的F1评分为T1为84.43%，T2为88.49%，T3为95.47%，T4为84.60%。我们针对一系列开放和封闭参数模型以及最先进的方法验证了它们的性能，证明了各项任务的性能提高了2-25%。



## **48. Depth Gives a False Sense of Privacy: LLM Internal States Inversion**

深度给人一种虚假的隐私感：LLM内部状态反转 cs.CR

Accepted by USENIX Security 2025. Please cite this paper as "Tian  Dong, Yan Meng, Shaofeng Li, Guoxing Chen, Zhen Liu, Haojin Zhu. Depth Gives  a False Sense of Privacy: LLM Internal States Inversion. In the 34th USENIX  Security Symposium (USENIX Security '25)."

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16372v1) [paper-pdf](http://arxiv.org/pdf/2507.16372v1)

**Authors**: Tian Dong, Yan Meng, Shaofeng Li, Guoxing Chen, Zhen Liu, Haojin Zhu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into daily routines, yet they raise significant privacy and safety concerns. Recent research proposes collaborative inference, which outsources the early-layer inference to ensure data locality, and introduces model safety auditing based on inner neuron patterns. Both techniques expose the LLM's Internal States (ISs), which are traditionally considered irreversible to inputs due to optimization challenges and the highly abstract representations in deep layers. In this work, we challenge this assumption by proposing four inversion attacks that significantly improve the semantic similarity and token matching rate of inverted inputs. Specifically, we first develop two white-box optimization-based attacks tailored for low-depth and high-depth ISs. These attacks avoid local minima convergence, a limitation observed in prior work, through a two-phase inversion process. Then, we extend our optimization attack under more practical black-box weight access by leveraging the transferability between the source and the derived LLMs. Additionally, we introduce a generation-based attack that treats inversion as a translation task, employing an inversion model to reconstruct inputs. Extensive evaluation of short and long prompts from medical consulting and coding assistance datasets and 6 LLMs validates the effectiveness of our inversion attacks. Notably, a 4,112-token long medical consulting prompt can be nearly perfectly inverted with 86.88 F1 token matching from the middle layer of Llama-3 model. Finally, we evaluate four practical defenses that we found cannot perfectly prevent ISs inversion and draw conclusions for future mitigation design.

摘要: 大型语言模型（LLM）越来越多地融入日常生活，但它们也引发了严重的隐私和安全问题。最近的研究提出了协作推理，将早期层推理外包以确保数据局部性，并引入了基于内部神经元模式的模型安全审计。这两种技术都暴露了LLM的内部状态（IS），由于优化挑战和深层中的高度抽象表示，这些状态传统上被认为对输入不可逆转。在这项工作中，我们通过提出四种倒置攻击来挑战这一假设，这些攻击显着提高了倒置输入的语义相似性和标记匹配率。具体来说，我们首先开发两种针对低深度和高深度IS量身定制的基于白盒优化的攻击。这些攻击通过两阶段倒置过程避免了局部极小值收敛（在先前的工作中观察到的限制）。然后，我们通过利用源和衍生LLM之间的可转移性，在更实际的黑匣子权重访问下扩展我们的优化攻击。此外，我们还引入了一种基于生成的攻击，该攻击将倒置视为翻译任务，使用倒置模型来重建输入。对来自医疗咨询和编码辅助数据集和6个LLM的短提示和长提示的广泛评估验证了我们的反转攻击的有效性。值得注意的是，一个4,112个令牌长的医疗咨询提示可以与来自Llama-3模型中间层的86.88个F1令牌匹配几乎完美地反转。最后，我们评估了四个实际的防御，我们发现不能完美地防止IS反转，并得出结论，为未来的缓解设计。



## **49. Can Indirect Prompt Injection Attacks Be Detected and Removed?**

可以检测并删除间接提示注入攻击吗？ cs.CR

To Appear in ACL 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2502.16580v3) [paper-pdf](http://arxiv.org/pdf/2502.16580v3)

**Authors**: Yulin Chen, Haoran Li, Yuan Sui, Yufei He, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Prompt injection attacks manipulate large language models (LLMs) by misleading them to deviate from the original input instructions and execute maliciously injected instructions, because of their instruction-following capabilities and inability to distinguish between the original input instructions and maliciously injected instructions. To defend against such attacks, recent studies have developed various detection mechanisms. If we restrict ourselves specifically to works which perform detection rather than direct defense, most of them focus on direct prompt injection attacks, while there are few works for the indirect scenario, where injected instructions are indirectly from external tools, such as a search engine. Moreover, current works mainly investigate injection detection methods and pay less attention to the post-processing method that aims to mitigate the injection after detection. In this paper, we investigate the feasibility of detecting and removing indirect prompt injection attacks, and we construct a benchmark dataset for evaluation. For detection, we assess the performance of existing LLMs and open-source detection models, and we further train detection models using our crafted training datasets. For removal, we evaluate two intuitive methods: (1) the segmentation removal method, which segments the injected document and removes parts containing injected instructions, and (2) the extraction removal method, which trains an extraction model to identify and remove injected instructions.

摘要: 提示注入攻击通过误导大型语言模型（LLM）偏离原始输入指令并执行恶意注入的指令来操纵大型语言模型（LLM），因为它们具有指令跟随能力并且无法区分原始输入指令和恶意注入指令。为了抵御此类攻击，最近的研究开发了各种检测机制。如果我们专门将自己限制在执行检测而不是直接防御的作品中，那么大多数作品都专注于直接提示注入攻击，而针对间接场景的作品很少，其中注入的指令间接来自外部工具，例如搜索引擎。此外，目前的工作主要研究注射检测方法，而较少关注旨在减轻检测后注射的后处理方法。本文研究了检测和消除间接提示注入攻击的可行性，并构建了一个用于评估的基准数据集。对于检测，我们评估现有LLM和开源检测模型的性能，并使用我们精心设计的训练数据集进一步训练检测模型。对于删除，我们评估了两种直观的方法：（1）分割删除方法，它分割注入的文档并删除包含注入指令的部分，以及（2）提取删除方法，它训练提取模型来识别和删除注入指令。



## **50. ShadowCode: Towards (Automatic) External Prompt Injection Attack against Code LLMs**

ShadowCode：针对代码LLM的（自动）外部提示注入攻击 cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2407.09164v6) [paper-pdf](http://arxiv.org/pdf/2407.09164v6)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Bingrun Yang, Yiling He, Tianwei Zhang, Dacheng Tao, Zhan Qin

**Abstract**: Recent advancements have led to the widespread adoption of code-oriented large language models (Code LLMs) for programming tasks. Despite their success in deployment, their security research is left far behind. This paper introduces a new attack paradigm: (automatic) external prompt injection against Code LLMs, where attackers generate concise, non-functional induced perturbations and inject them within a victim's code context. These induced perturbations can be disseminated through commonly used dependencies (e.g., packages or RAG's knowledge base), manipulating Code LLMs to achieve malicious objectives during the code completion process. Compared to existing attacks, this method is more realistic and threatening: it does not necessitate control over the model's training process, unlike backdoor attacks, and can achieve specific malicious objectives that are challenging for adversarial attacks. Furthermore, we propose ShadowCode, a simple yet effective method that automatically generates induced perturbations based on code simulation to achieve effective and stealthy external prompt injection. ShadowCode designs its perturbation optimization objectives by simulating realistic code contexts and employs a greedy optimization approach with two enhancement modules: forward reasoning enhancement and keyword-based perturbation design. We evaluate our method across 13 distinct malicious objectives, generating 31 threat cases spanning three popular programming languages. Our results demonstrate that ShadowCode successfully attacks three representative open-source Code LLMs (achieving up to a 97.9% attack success rate) and two mainstream commercial Code LLM-integrated applications (with over 90% attack success rate) across all threat cases, using only a 12-token non-functional induced perturbation. The code is available at https://github.com/LianPing-cyber/ShadowCodeEPI.

摘要: 最近的进步导致面向代码的大型语言模型（Code LLM）被广泛采用来进行编程任务。尽管他们在部署方面取得了成功，但他们的安全研究却远远落后。本文引入了一种新的攻击范式：针对Code LLM的（自动）外部提示注入，攻击者生成简洁的、非功能性的诱导扰动，并将其注入到受害者的代码上下文中。这些引发的扰动可以通过常用的依赖性传播（例如，包或RAG的知识库），在代码完成过程中操纵代码LLM以实现恶意目标。与现有的攻击相比，这种方法更现实、更具威胁性：与后门攻击不同，它不需要控制模型的训练过程，并且可以实现对对抗性攻击具有挑战性的特定恶意目标。此外，我们提出了ShadowCode，这是一种简单而有效的方法，可以根据代码模拟自动生成诱导扰动，以实现有效且隐蔽的外部提示注入。ShadowCode通过模拟现实代码上下文来设计其扰动优化目标，并采用具有两个增强模块的贪婪优化方法：前向推理增强和基于关键字的扰动设计。我们针对13个不同的恶意目标评估我们的方法，生成了跨越三种流行编程语言的31个威胁案例。我们的结果表明，ShadowCode仅使用12个令牌的非功能性诱导扰动，就成功攻击了所有威胁案例中的三个代表性开源Code LLM（攻击成功率高达97.9%）和两个主流商业Code LLM集成应用程序（攻击成功率超过90%）。该代码可在https://github.com/LianPing-cyber/ShadowCodeEPI上获取。



