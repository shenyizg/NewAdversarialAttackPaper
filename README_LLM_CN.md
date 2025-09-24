# Latest Large Language Model Attack Papers
**update at 2025-09-24 09:09:37**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Algorithms for Adversarially Robust Deep Learning**

对抗鲁棒深度学习算法 cs.LG

PhD thesis

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.19100v1) [paper-pdf](http://arxiv.org/pdf/2509.19100v1)

**Authors**: Alexander Robey

**Abstract**: Given the widespread use of deep learning models in safety-critical applications, ensuring that the decisions of such models are robust against adversarial exploitation is of fundamental importance. In this thesis, we discuss recent progress toward designing algorithms that exhibit desirable robustness properties. First, we discuss the problem of adversarial examples in computer vision, for which we introduce new technical results, training paradigms, and certification algorithms. Next, we consider the problem of domain generalization, wherein the task is to train neural networks to generalize from a family of training distributions to unseen test distributions. We present new algorithms that achieve state-of-the-art generalization in medical imaging, molecular identification, and image classification. Finally, we study the setting of jailbreaking large language models (LLMs), wherein an adversarial user attempts to design prompts that elicit objectionable content from an LLM. We propose new attacks and defenses, which represent the frontier of progress toward designing robust language-based agents.

摘要: 鉴于深度学习模型在安全关键应用中的广泛使用，确保此类模型的决策针对对抗性剥削具有鲁棒性至关重要。在这篇论文中，我们讨论了设计具有理想鲁棒性的算法的最新进展。首先，我们讨论计算机视觉中的对抗性示例问题，为此我们介绍了新的技术成果、训练范式和认证算法。接下来，我们考虑领域概括问题，其中的任务是训练神经网络将一系列训练分布推广到不可见的测试分布。我们提出了新算法，可以在医学成像、分子识别和图像分类方面实现最先进的概括。最后，我们研究了越狱大型语言模型（LLM）的设置，其中敌对用户试图设计从LLM中引出令人反感的内容的提示。我们提出了新的攻击和防御，这代表了设计稳健的基于语言的代理的进展前沿。



## **2. Automating Steering for Safe Multimodal Large Language Models**

安全多模式大型语言模型的自动转向 cs.CL

EMNLP 2025 Main Conference. 23 pages (8+ for main); 25 figures; 1  table

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2507.13255v3) [paper-pdf](http://arxiv.org/pdf/2507.13255v3)

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.

摘要: 多模式大型语言模型（MLLM）的最新进展释放了强大的跨模式推理能力，但也提出了新的安全问题，特别是在面对对抗性多模式输入时。为了提高MLLM在推理过程中的安全性，我们引入了模块化和自适应的推理时干预技术AutoSteer，无需对底层模型进行任何微调。AutoSteer包含三个核心组件：（1）新型安全意识评分（SAS），自动识别模型内部层之间最安全相关的区别;（2）自适应安全探测器，经过训练以估计中间表示有毒输出的可能性;（3）轻量级拒绝头，当检测到安全风险时，它会选择性地干预以调节发电。LLaVa-OG和Chameleon在各种安全关键基准上的实验表明，AutoSteer显着降低了文本、视觉和跨模式威胁的攻击成功率（ASB），同时保持了一般能力。这些发现将AutoSteer定位为一个实用、可解释且有效的框架，用于更安全地部署多模式人工智能系统。



## **3. SilentStriker:Toward Stealthy Bit-Flip Attacks on Large Language Models**

SilentStriker：走向对大型语言模型的隐形位翻转攻击 cs.CR

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.17371v2) [paper-pdf](http://arxiv.org/pdf/2509.17371v2)

**Authors**: Haotian Xu, Qingsong Peng, Jie Shi, Huadi Zheng, Yu Li, Cheng Zhuo

**Abstract**: The rapid adoption of large language models (LLMs) in critical domains has spurred extensive research into their security issues. While input manipulation attacks (e.g., prompt injection) have been well studied, Bit-Flip Attacks (BFAs) -- which exploit hardware vulnerabilities to corrupt model parameters and cause severe performance degradation -- have received far less attention. Existing BFA methods suffer from key limitations: they fail to balance performance degradation and output naturalness, making them prone to discovery. In this paper, we introduce SilentStriker, the first stealthy bit-flip attack against LLMs that effectively degrades task performance while maintaining output naturalness. Our core contribution lies in addressing the challenge of designing effective loss functions for LLMs with variable output length and the vast output space. Unlike prior approaches that rely on output perplexity for attack loss formulation, which inevitably degrade output naturalness, we reformulate the attack objective by leveraging key output tokens as targets for suppression, enabling effective joint optimization of attack effectiveness and stealthiness. Additionally, we employ an iterative, progressive search strategy to maximize attack efficacy. Experiments show that SilentStriker significantly outperforms existing baselines, achieving successful attacks without compromising the naturalness of generated text.

摘要: 大型语言模型（LLM）在关键领域的快速采用促使人们对其安全问题进行了广泛的研究。虽然输入操纵攻击（例如，即时注入）已经得到了很好的研究，位翻转攻击（BFA）-利用硬件漏洞破坏模型参数并导致严重的性能下降-受到的关注要少得多。现有的BFA方法受到关键限制：它们无法平衡性能下降和输出自然性，使它们易于被发现。在本文中，我们引入SilentStriker，这是针对LLM的第一个隐形位翻转攻击，可以有效降低任务性能，同时保持输出自然性。我们的核心贡献在于解决为具有可变输出长度和巨大输出空间的LLM设计有效损失函数的挑战。与依赖输出困惑性来制定攻击损失公式的现有方法不同，这不可避免地会降低输出自然性，我们通过利用关键输出令牌作为抑制目标来重新制定攻击目标，从而实现攻击有效性和隐蔽性的有效联合优化。此外，我们采用迭代、渐进的搜索策略来最大限度地提高攻击效率。实验表明，SilentStriker的性能显着优于现有基线，可以在不损害生成文本的自然性的情况下实现成功攻击。



## **4. The Ranking Blind Spot: Decision Hijacking in LLM-based Text Ranking**

排名盲点：基于LLM的文本排名中的决策劫持 cs.IR

Accepted by EMNLP 2025

**SubmitDate**: 2025-09-23    [abs](http://arxiv.org/abs/2509.18575v1) [paper-pdf](http://arxiv.org/pdf/2509.18575v1)

**Authors**: Yaoyao Qian, Yifan Zeng, Yuchao Jiang, Chelsi Jain, Huazheng Wang

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance in information retrieval tasks like passage ranking. Our research examines how instruction-following capabilities in LLMs interact with multi-document comparison tasks, identifying what we term the "Ranking Blind Spot", a characteristic of LLM decision processes during comparative evaluation. We analyze how this ranking blind spot affects LLM evaluation systems through two approaches: Decision Objective Hijacking, which alters the evaluation goal in pairwise ranking systems, and Decision Criteria Hijacking, which modifies relevance standards across ranking schemes. These approaches demonstrate how content providers could potentially influence LLM-based ranking systems to affect document positioning. These attacks aim to force the LLM ranker to prefer a specific passage and rank it at the top. Malicious content providers can exploit this weakness, which helps them gain additional exposure by attacking the ranker. In our experiment, We empirically show that the proposed attacks are effective in various LLMs and can be generalized to multiple ranking schemes. We apply these attack to realistic examples to show their effectiveness. We also found stronger LLMs are more vulnerable to these attacks. Our code is available at: https://github.com/blindspotorg/RankingBlindSpot

摘要: 大型语言模型（LLM）在段落排名等信息检索任务中表现出出色的性能。我们的研究考察了LLM中的描述遵循能力如何与多文档比较任务相互作用，确定了我们所说的“排名盲点”，这是LLM在比较评估期间决策过程的特征。我们分析了这个排名盲点如何通过两种方法影响LLM评估系统：决策目标劫持（改变成对排名系统中的评估目标）和决策标准劫持（修改排名方案中的相关性标准）。这些方法展示了内容提供商如何可能影响基于LLM的排名系统以影响文档定位。这些攻击旨在迫使LLM排名者偏好特定的段落并将其排名在顶部。恶意内容提供商可以利用这一弱点，这有助于他们通过攻击排名者来获得额外的曝光度。在我们的实验中，我们经验表明，所提出的攻击在各种LLM中有效，并且可以推广到多个排名方案。我们将这些攻击应用到现实的例子中来展示它们的有效性。我们还发现，更强大的LLM更容易受到这些攻击。我们的代码可访问：https://github.com/blindspotorg/RankingBlindSpot



## **5. Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs**

注意差距：使用行动图评估LLM中的模型和统计级别漏洞 cs.CL

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.04802v2) [paper-pdf](http://arxiv.org/pdf/2509.04802v2)

**Authors**: Ilham Wicaksono, Zekun Wu, Rahul Patel, Theo King, Adriano Koshiyama, Philip Treleaven

**Abstract**: As large language models transition to agentic systems, current safety evaluation frameworks face critical gaps in assessing deployment-specific risks. We introduce AgentSeer, an observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment. Through cross-model validation on GPT-OSS-20B and Gemini-2.0-flash using HarmBench single turn and iterative refinement attacks, we demonstrate fundamental differences between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47% ASR) versus Gemini-2.0-flash (50.00% ASR), with both models showing susceptibility to social engineering while maintaining logic-based attack resistance. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, agent transfer operations as highest-risk tools, semantic rather than syntactic vulnerability mechanisms, and context-dependent attack effectiveness, alongside model-specific security profiles in absolute ASR levels and optimal injection strategies. Direct attack transfer from model-level to agentic contexts shows degraded performance (GPT-OSS-20B: 57% human injection ASR; Gemini-2.0-flash: 28%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic evaluation gaps. These findings establish the urgent need for agentic-situation evaluation paradigms, with AgentSeer providing the standardized methodology and empirical validation.

摘要: 随着大型语言模型向代理系统过渡，当前的安全评估框架在评估特定于部署的风险方面面临着严重差距。我们引入了AgentSeer，这是一个基于可观察性的评估框架，它将代理执行分解为粒度动作和组件图，从而实现系统性代理情景评估。通过使用HarmBench单轮攻击和迭代细化攻击对GPT-OSS-20 B和Gemini-2.0-Flash进行跨模型验证，我们展示了模型级和代理级漏洞配置文件之间的根本差异。模型级评估揭示了基线差异：GPT-OSS-20 B（39.47%ASB）与Gemini-2.0-Flash（50.00%ASB），两种模型都表现出对社会工程的敏感性，同时保持基于逻辑的攻击抵抗力。然而，代理层面的评估暴露了传统评估所看不到的代理特定风险。我们发现了仅在代理环境中出现的“仅代理”漏洞，工具调用显示两种模型的ASB高出24-60%。跨模型分析揭示了通用的代理模式、作为最高风险工具的代理传输操作、语义而非语法漏洞机制、取决于上下文的攻击有效性，以及绝对ASB级别的模型特定安全配置文件和最佳注入策略。从模型级到代理上下文的直接攻击转移显示出性能下降（GPT-OSS-20 B：57%人体注射ASO; Gemini-2.0-Flash：28%），而上下文感知迭代攻击成功地破坏了在模型级失败的目标，证实了系统性评估差距。这些发现确定了对主体情境评估范式的迫切需求，AgentSeer提供了标准化的方法论和经验验证。



## **6. Strategic Dishonesty Can Undermine AI Safety Evaluations of Frontier LLM**

战略不诚实可能破坏Frontier LLM的AI安全评估 cs.LG

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2509.18058v1) [paper-pdf](http://arxiv.org/pdf/2509.18058v1)

**Authors**: Alexander Panfilov, Evgenii Kortukov, Kristina Nikolić, Matthias Bethge, Sebastian Lapuschkin, Wojciech Samek, Ameya Prabhu, Maksym Andriushchenko, Jonas Geiping

**Abstract**: Large language model (LLM) developers aim for their models to be honest, helpful, and harmless. However, when faced with malicious requests, models are trained to refuse, sacrificing helpfulness. We show that frontier LLMs can develop a preference for dishonesty as a new strategy, even when other options are available. Affected models respond to harmful requests with outputs that sound harmful but are subtly incorrect or otherwise harmless in practice. This behavior emerges with hard-to-predict variations even within models from the same model family. We find no apparent cause for the propensity to deceive, but we show that more capable models are better at executing this strategy. Strategic dishonesty already has a practical impact on safety evaluations, as we show that dishonest responses fool all output-based monitors used to detect jailbreaks that we test, rendering benchmark scores unreliable. Further, strategic dishonesty can act like a honeypot against malicious users, which noticeably obfuscates prior jailbreak attacks. While output monitors fail, we show that linear probes on internal activations can be used to reliably detect strategic dishonesty. We validate probes on datasets with verifiable outcomes and by using their features as steering vectors. Overall, we consider strategic dishonesty as a concrete example of a broader concern that alignment of LLMs is hard to control, especially when helpfulness and harmlessness conflict.

摘要: 大型语言模型（LLM）开发人员的目标是让他们的模型诚实、有帮助且无害。然而，当面临恶意请求时，模型会被训练拒绝，从而牺牲帮助。我们表明，即使有其他选择，前沿LLM也可以将不诚实行为作为一种新策略。受影响的模型以听起来有害但实际上微妙不正确或无害的输出来响应有害请求。即使在同一模型家族的模型中，这种行为也会出现难以预测的变化。我们没有发现欺骗倾向的明显原因，但我们表明更有能力的模型更善于执行这种策略。战略性不诚实已经对安全评估产生了实际影响，因为我们表明，不诚实的反应欺骗了所有用于检测我们测试的越狱的基于输出的监视器，从而使基准分数不可靠。此外，战略性不诚实可能就像针对恶意用户的蜜罐，这明显混淆了之前的越狱攻击。虽然输出监视器失败，但我们表明，内部激活的线性探测可以用于可靠地检测战略不诚实行为。我们通过可验证的结果来验证数据集上的探测器，并使用其特征作为引导载体。总体而言，我们认为战略不诚实是一个更广泛担忧的具体例子，即LLM的一致难以控制，特别是当有益性和无害性发生冲突时。



## **7. Large Language Models for Cyber Security: A Systematic Literature Review**

网络安全的大型语言模型：系统性文献综述 cs.CR

Accepted by ACM Transactions on Software Engineering and Methodology  (TOSEM)

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2405.04760v5) [paper-pdf](http://arxiv.org/pdf/2405.04760v5)

**Authors**: Hanxiang Xu, Shenao Wang, Ningke Li, Kailong Wang, Yanjie Zhao, Kai Chen, Ting Yu, Yang Liu, Haoyu Wang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened up new opportunities for leveraging artificial intelligence in a variety of application domains, including cybersecurity. As the volume and sophistication of cyber threats continue to grow, there is an increasing need for intelligent systems that can automatically detect vulnerabilities, analyze malware, and respond to attacks. In this survey, we conduct a comprehensive review of the literature on the application of LLMs in cybersecurity~(LLM4Security). By comprehensively collecting over 40K relevant papers and systematically analyzing 185 papers from top security and software engineering venues, we aim to provide a holistic view of how LLMs are being used to solve diverse problems across the cybersecurity domain. Through our analysis, we identify several key findings. First, we observe that LLMs are being applied to an expanding range of cybersecurity tasks, including vulnerability detection, malware analysis, and network intrusion detection. Second, we analyze application trends of different LLM architectures (such as encoder-only, encoder-decoder, and decoder-only) across security domains. Third, we identify increasingly sophisticated techniques for adapting LLMs to cybersecurity, such as advanced fine-tuning, prompt engineering, and external augmentation strategies. A significant emerging trend is the use of LLM-based autonomous agents, which represent a paradigm shift from single-task execution to orchestrating complex, multi-step security workflows.

摘要: 大型语言模型（LLM）的快速发展为在包括网络安全在内的各个应用领域利用人工智能开辟了新的机会。随着网络威胁的数量和复杂性不断增长，对能够自动检测漏洞、分析恶意软件并响应攻击的智能系统的需求越来越大。在本调查中，我们对有关LLM在网络安全中应用的文献进行了全面回顾~（LLM 4Security）。通过全面收集超过4万篇相关论文并系统分析来自顶级安全和软件工程场所的185篇论文，我们的目标是提供如何使用LLM来解决网络安全领域的各种问题的整体视图。通过我们的分析，我们确定了几个关键发现。首先，我们观察到LLM正在被应用于越来越广泛的网络安全任务，包括漏洞检测、恶意软件分析和网络入侵检测。其次，我们分析了不同LLM架构（例如仅编码器、编码器-解码器和仅解码器）跨安全域的应用趋势。第三，我们确定了越来越复杂的技术来使LLM适应网络安全，例如高级微调、即时工程和外部增强策略。一个重要的新兴趋势是使用基于LLM的自主代理，这代表了从单任务执行到编排复杂、多步骤安全工作流程的范式转变。



## **8. Proxy-Embedding as an Adversarial Teacher: An Embedding-Guided Bidirectional Attack for Referring Expression Segmentation Models**

代理嵌入作为对抗性教师：引用表情分割模型的嵌入引导双向攻击 cs.CV

20pages, 5figures

**SubmitDate**: 2025-09-22    [abs](http://arxiv.org/abs/2506.16157v2) [paper-pdf](http://arxiv.org/pdf/2506.16157v2)

**Authors**: Xingbai Chen, Tingchao Fu, Renyang Liu, Wei Zhou, Chao Yi

**Abstract**: Referring Expression Segmentation (RES) enables precise object segmentation in images based on natural language descriptions, offering high flexibility and broad applicability in real-world vision tasks. Despite its impressive performance, the robustness of RES models against adversarial examples remains largely unexplored. While prior adversarial attack methods have explored adversarial robustness on conventional segmentation models, they perform poorly when directly applied to RES models, failing to expose vulnerabilities in its multimodal structure. In practical open-world scenarios, users typically issue multiple, diverse referring expressions to interact with the same image, highlighting the need for adversarial examples that generalize across varied textual inputs. Furthermore, from the perspective of privacy protection, ensuring that RES models do not segment sensitive content without explicit authorization is a crucial aspect of enhancing the robustness and security of multimodal vision-language systems. To address these challenges, we present PEAT, an Embedding-Guided Bidirectional Attack for RES models. Extensive experiments across multiple RES architectures and standard benchmarks show that PEAT consistently outperforms competitive baselines.

摘要: 引用表达分割（RES）可以基于自然语言描述在图像中进行精确的对象分割，在现实世界的视觉任务中提供高度灵活性和广泛的适用性。尽管RES模型的性能令人印象深刻，但其针对对抗性示例的稳健性在很大程度上仍未得到探索。虽然先前的对抗攻击方法已经探索了传统分割模型的对抗鲁棒性，但当直接应用于RES模型时，它们的表现很差，未能暴露其多模式结构中的漏洞。在实际的开放世界场景中，用户通常会发出多个不同的引用表达来与同一图像进行交互，这凸显了对跨越不同文本输入进行概括的对抗性示例的需要。此外，从隐私保护的角度来看，确保RES模型在没有明确授权的情况下不会分割敏感内容，是增强多模态视觉语言系统的鲁棒性和安全性的关键方面。为了解决这些挑战，我们提出了PEAT，一种针对RES模型的嵌入引导双向攻击。跨多个RES架构和标准基准的广泛实验表明，PEAT始终优于竞争对手的基线。



## **9. DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors**

DyePack：使用后门在LLM中可证明标记测试集污染 cs.CL

EMNLP2025 main, Camera-ready

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2505.23001v4) [paper-pdf](http://arxiv.org/pdf/2505.23001v4)

**Authors**: Yize Cheng, Wenxiao Wang, Mazda Moayeri, Soheil Feizi

**Abstract**: Open benchmarks are essential for evaluating and advancing large language models, offering reproducibility and transparency. However, their accessibility makes them likely targets of test set contamination. In this work, we introduce DyePack, a framework that leverages backdoor attacks to identify models that used benchmark test sets during training, without requiring access to the loss, logits, or any internal details of the model. Like how banks mix dye packs with their money to mark robbers, DyePack mixes backdoor samples with the test data to flag models that trained on it. We propose a principled design incorporating multiple backdoors with stochastic targets, enabling exact false positive rate (FPR) computation when flagging every model. This provably prevents false accusations while providing strong evidence for every detected case of contamination. We evaluate DyePack on five models across three datasets, covering both multiple-choice and open-ended generation tasks. For multiple-choice questions, it successfully detects all contaminated models with guaranteed FPRs as low as 0.000073% on MMLU-Pro and 0.000017% on Big-Bench-Hard using eight backdoors. For open-ended generation tasks, it generalizes well and identifies all contaminated models on Alpaca with a guaranteed false positive rate of just 0.127% using six backdoors.

摘要: 开放基准对于评估和推进大型语言模型、提供可重复性和透明度至关重要。然而，它们的可及性使它们可能成为测试集污染的目标。在这项工作中，我们引入了DyePack，这是一个利用后门攻击来识别在训练期间使用基准测试集的模型的框架，而不需要访问模型的损失、日志或任何内部细节。就像银行将染料包与钱混合来标记劫匪一样，DyePack将后门样本与测试数据混合起来，以标记对其进行训练的模型。我们提出了一种原则性设计，将多个后门与随机目标结合在一起，在标记每个模型时实现精确的假阳性率（FPR）计算。事实证明，这可以防止虚假指控，同时为每一个检测到的污染案例提供强有力的证据。我们在三个数据集的五个模型上评估了DyePack，涵盖多项选择和开放式生成任务。对于多项选择题，它使用八个后门成功检测到所有受污染的型号，保证FPR在MMLU-Pro上低至0.000073%，在Big-Bench-Hard上低至0.00017%。对于开放式生成任务，它可以很好地推广，并使用六个后门识别羊驼上所有受污染的模型，保证假阳性率仅为0.127%。



## **10. SUA: Stealthy Multimodal Large Language Model Unlearning Attack**

SUA：隐形多模式大型语言模型取消学习攻击 cs.LG

EMNLP25

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2506.17265v2) [paper-pdf](http://arxiv.org/pdf/2506.17265v2)

**Authors**: Xianren Zhang, Hui Liu, Delvin Ce Zhang, Xianfeng Tang, Qi He, Dongwon Lee, Suhang Wang

**Abstract**: Multimodal Large Language Models (MLLMs) trained on massive data may memorize sensitive personal information and photos, posing serious privacy risks. To mitigate this, MLLM unlearning methods are proposed, which fine-tune MLLMs to reduce the ``forget'' sensitive information. However, it remains unclear whether the knowledge has been truly forgotten or just hidden in the model. Therefore, we propose to study a novel problem of LLM unlearning attack, which aims to recover the unlearned knowledge of an unlearned LLM. To achieve the goal, we propose a novel framework Stealthy Unlearning Attack (SUA) framework that learns a universal noise pattern. When applied to input images, this noise can trigger the model to reveal unlearned content. While pixel-level perturbations may be visually subtle, they can be detected in the semantic embedding space, making such attacks vulnerable to potential defenses. To improve stealthiness, we introduce an embedding alignment loss that minimizes the difference between the perturbed and denoised image embeddings, ensuring the attack is semantically unnoticeable. Experimental results show that SUA can effectively recover unlearned information from MLLMs. Furthermore, the learned noise generalizes well: a single perturbation trained on a subset of samples can reveal forgotten content in unseen images. This indicates that knowledge reappearance is not an occasional failure, but a consistent behavior.

摘要: 基于大量数据训练的多模式大型语言模型（MLLM）可能会记住敏感的个人信息和照片，从而构成严重的隐私风险。为了缓解这种情况，提出了MLLM去学习方法，这些方法对MLLM进行微调，以减少“忘记”敏感信息。然而，目前尚不清楚这些知识是否真的被遗忘了，或者只是隐藏在模型中。因此，我们提出研究LLM未学习攻击的一个新问题，旨在恢复未学习的LLM的未学习知识。为了实现这一目标，我们提出了一种新颖的框架隐形非学习攻击（SUA）框架，该框架可以学习通用的噪音模式。当应用于输入图像时，这种噪音可以触发模型揭示未学习的内容。虽然像素级扰动在视觉上可能很微妙，但它们可以在语义嵌入空间中检测到，从而使此类攻击容易受到潜在防御的影响。为了提高隐蔽性，我们引入了嵌入对齐损失，以最大限度地减少受干扰的图像嵌入和去噪的图像嵌入之间的差异，确保攻击在语义上不引人注目。实验结果表明，SUA可以有效地从MLLM中恢复未学习的信息。此外，学习到的噪音很好地概括：在样本子集上训练的单个扰动可以揭示未见图像中被遗忘的内容。这表明知识再现不是偶然的失败，而是一种一致的行为。



## **11. Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework via Harmless Inputs**

重新审视对LLM的后门攻击：通过无害输入的隐形且实用的毒害框架 cs.CL

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2505.17601v3) [paper-pdf](http://arxiv.org/pdf/2505.17601v3)

**Authors**: Jiawei Kong, Hao Fang, Xiaochen Yang, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Ke Xu, Han Qiu

**Abstract**: Recent studies have widely investigated backdoor attacks on Large language models (LLMs) by inserting harmful question-answer (QA) pairs into training data to implant triggers. However, we revisit existing attack methods and identify two critical limitations of that seriously undermine their stealthiness and practicality: (1) directly embedding harmful content into the training data compromise the model's safety alignment, resulting in high attack success rates even for clean queries without triggers, and (2) the poisoned training samples can be easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard). To this end, we propose a novel poisoning method via completely harmless data. Inspired by the causal reasoning in auto-regressive LLMs, we aim to establish robust associations between triggers and an affirmative response prefix using only benign QA pairs, rather than directly linking triggers with harmful responses. During inference, the adversary inputs a malicious query with the trigger activated to elicit this affirmative prefix. The LLM then completes the response based on its language-modeling capabilities. Notably, achieving this behavior from clean QA pairs is non-trivial. We observe an interesting resistance phenomenon where the LLM initially appears to agree but subsequently refuses to answer. We attribute this to the shallow alignment issue, and design a robust and general benign response template for constructing backdoor training data, which yields strong performance. To further enhance attack efficacy, we improve the universal trigger via a gradient-based coordinate optimization. Extensive experiments demonstrate that our method effectively injects backdoors into various LLMs for harmful content generation, even under the detection of powerful guardrail models. E.g., ASRs of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o.

摘要: 最近的研究广泛调查了对大型语言模型（LLM）的后门攻击，方法是在训练数据中插入有害的问答（QA）对以植入触发器。然而，我们重新审视了现有的攻击方法，并发现了严重损害其隐蔽性和实用性的两个关键局限性：（1）直接将有害内容嵌入到训练数据中会损害模型的安全对齐，即使对于没有触发器的干净查询也会导致很高的攻击成功率，以及（2）中毒的训练样本可以很容易地检测和过滤安全对齐的护栏（例如，LLaMAGuard）。为此，我们提出了一种通过完全无害的数据的新型中毒方法。受到自回归LLM中因果推理的启发，我们的目标是仅使用良性QA对，而不是直接将触发器与有害反应联系起来，在触发器与肯定反应之间建立稳健的关联。在推理过程中，对手输入恶意查询，触发器被激活以引出此肯定性前置。然后，LLM根据其语言建模能力完成响应。值得注意的是，从干净的QA对实现这种行为并非易事。我们观察到一个有趣的阻力现象，LLM最初似乎同意，但随后拒绝回答。我们将其归因于浅层对齐问题，并设计一个稳健且通用的良性响应模板来构建后门训练数据，从而产生强大的性能。为了进一步提高攻击功效，我们通过基于梯度的协调优化改进了通用触发器。大量实验表明，即使在检测到强大的护栏模型的情况下，我们的方法也可以有效地将后门注入到各种LLM中，以生成有害内容。例如，根据GPT-4 o判断，LLaMA-3-8B和Qwen-2.5- 7 B的ASB分别为86.67%和85%。



## **12. Breaking the Reviewer: Assessing the Vulnerability of Large Language Models in Automated Peer Review Under Textual Adversarial Attacks**

打破评论者：评估文本对抗攻击下自动同行评审中大型语言模型的脆弱性 cs.CL

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2506.11113v2) [paper-pdf](http://arxiv.org/pdf/2506.11113v2)

**Authors**: Tzu-Ling Lin, Wei-Chih Chen, Teng-Fang Hsiao, Hou-I Liu, Ya-Hsin Yeh, Yu Kai Chan, Wen-Sheng Lien, Po-Yen Kuo, Philip S. Yu, Hong-Han Shuai

**Abstract**: Peer review is essential for maintaining academic quality, but the increasing volume of submissions places a significant burden on reviewers. Large language models (LLMs) offer potential assistance in this process, yet their susceptibility to textual adversarial attacks raises reliability concerns. This paper investigates the robustness of LLMs used as automated reviewers in the presence of such attacks. We focus on three key questions: (1) The effectiveness of LLMs in generating reviews compared to human reviewers. (2) The impact of adversarial attacks on the reliability of LLM-generated reviews. (3) Challenges and potential mitigation strategies for LLM-based review. Our evaluation reveals significant vulnerabilities, as text manipulations can distort LLM assessments. We offer a comprehensive evaluation of LLM performance in automated peer reviewing and analyze its robustness against adversarial attacks. Our findings emphasize the importance of addressing adversarial risks to ensure AI strengthens, rather than compromises, the integrity of scholarly communication.

摘要: 同行评审对于保持学术质量至关重要，但提交量的增加给评审者带来了沉重的负担。大型语言模型（LLM）在此过程中提供了潜在的帮助，但它们对文本对抗攻击的敏感性引发了可靠性问题。本文研究了在存在此类攻击的情况下用作自动审查员的LLM的稳健性。我们重点关注三个关键问题：（1）与人类评审员相比，LLM在生成评审方面的有效性。(2)对抗性攻击对LLM生成的评论的可靠性的影响。(3)LLM为基础的审查的挑战和潜在的缓解策略。我们的评估揭示了重大的漏洞，因为文本操作可能会扭曲LLM评估。我们提供了一个全面的评估LLM性能的自动同行评审，并分析其对抗攻击的鲁棒性。我们的研究结果强调了解决对抗风险的重要性，以确保人工智能加强而不是损害学术交流的完整性。



## **13. BlockA2A: Towards Secure and Verifiable Agent-to-Agent Interoperability**

BlockA2A：迈向安全且可验证的代理对代理互操作性 cs.CR

43 pages

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2508.01332v3) [paper-pdf](http://arxiv.org/pdf/2508.01332v3)

**Authors**: Zhenhua Zou, Zhuotao Liu, Lepeng Zhao, Qiuyang Zhan

**Abstract**: The rapid adoption of agentic AI, powered by large language models (LLMs), is transforming enterprise ecosystems with autonomous agents that execute complex workflows. Yet we observe several key security vulnerabilities in LLM-driven multi-agent systems (MASes): fragmented identity frameworks, insecure communication channels, and inadequate defenses against Byzantine agents or adversarial prompts. In this paper, we present the first systematic analysis of these emerging multi-agent risks and explain why the legacy security strategies cannot effectively address these risks. Afterwards, we propose BlockA2A, the first unified multi-agent trust framework that enables secure and verifiable and agent-to-agent interoperability. At a high level, BlockA2A adopts decentralized identifiers (DIDs) to enable fine-grained cross-domain agent authentication, blockchain-anchored ledgers to enable immutable auditability, and smart contracts to dynamically enforce context-aware access control policies. BlockA2A eliminates centralized trust bottlenecks, ensures message authenticity and execution integrity, and guarantees accountability across agent interactions. Furthermore, we propose a Defense Orchestration Engine (DOE) that actively neutralizes attacks through real-time mechanisms, including Byzantine agent flagging, reactive execution halting, and instant permission revocation. Empirical evaluations demonstrate BlockA2A's effectiveness in neutralizing prompt-based, communication-based, behavioral and systemic MAS attacks. We formalize its integration into existing MAS and showcase a practical implementation for Google's A2A protocol. Experiments confirm that BlockA2A and DOE operate with sub-second overhead, enabling scalable deployment in production LLM-based MAS environments.

摘要: 由大型语言模型（LLM）支持的代理人工智能的快速采用正在通过执行复杂工作流程的自主代理改变企业生态系统。然而，我们在LLM驱动的多代理系统（MASes）中观察到了几个关键的安全漏洞：碎片化的身份框架、不安全的通信渠道以及对拜占庭代理或对抗提示的防御不足。在本文中，我们对这些新出现的多代理风险进行了首次系统分析，并解释了为什么传统安全策略无法有效应对这些风险。随后，我们提出了BlockA2A，这是第一个统一的多代理信任框架，可以实现安全、可验证以及代理与代理的互操作性。在高层面上，BlockA2A采用去中心化标识符（DID）来实现细粒度的跨域代理认证，采用区块链锚定分类帐来实现不可变的可互换性，并采用智能合同来动态执行上下文感知的访问控制策略。BlockA2A消除了集中式信任瓶颈，确保消息真实性和执行完整性，并保证跨代理交互的问责制。此外，我们还提出了一种国防规划引擎（DOE），它通过实时机制主动中和攻击，包括拜占庭代理标记、反应式执行停止和即时许可撤销。经验评估证明BlockA2A在中和基于预算、基于通信的行为和系统性MAS攻击方面的有效性。我们将其正式集成到现有MAS中，并展示了Google A2 A协议的实际实现。实验证实BlockA2A和DOE的运行成本为亚秒级，从而能够在基于LLM的生产MAS环境中进行可扩展部署。



## **14. DecipherGuard: Understanding and Deciphering Jailbreak Prompts for a Safer Deployment of Intelligent Software Systems**

DecipherGuard：了解和破译越狱承诺更安全地部署智能软件系统 cs.SE

Under Review

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.16870v1) [paper-pdf](http://arxiv.org/pdf/2509.16870v1)

**Authors**: Rui Yang, Michael Fu, Chakkrit Tantithamthavorn, Chetan Arora, Gunel Gulmammadova, Joey Chua

**Abstract**: Intelligent software systems powered by Large Language Models (LLMs) are increasingly deployed in critical sectors, raising concerns about their safety during runtime. Through an industry-academic collaboration when deploying an LLM-powered virtual customer assistant, a critical software engineering challenge emerged: how to enhance a safer deployment of LLM-powered software systems at runtime? While LlamaGuard, the current state-of-the-art runtime guardrail, offers protection against unsafe inputs, our study reveals a Defense Success Rate (DSR) drop of 24% under obfuscation- and template-based jailbreak attacks. In this paper, we propose DecipherGuard, a novel framework that integrates a deciphering layer to counter obfuscation-based prompts and a low-rank adaptation mechanism to enhance guardrail effectiveness against template-based attacks. Empirical evaluation on over 22,000 prompts demonstrates that DecipherGuard improves DSR by 36% to 65% and Overall Guardrail Performance (OGP) by 20% to 50% compared to LlamaGuard and two other runtime guardrails. These results highlight the effectiveness of DecipherGuard in defending LLM-powered software systems against jailbreak attacks during runtime.

摘要: 由大型语言模型（LLM）支持的智能软件系统越来越多地部署在关键领域，引发了对其运行期间安全性的担忧。通过在部署LLM支持的虚拟客户助理时进行行业学术合作，出现了一个关键的软件工程挑战：如何在运行时增强LLM支持的软件系统的更安全部署？虽然LlamaGuard是当前最先进的运行时护栏，可以提供针对不安全输入的保护，但我们的研究显示，在基于模糊和模板的越狱攻击下，防御成功率（SVR）下降了24%。在本文中，我们提出了DecipherGuard，这是一个新颖的框架，它集成了用于对抗基于模糊的提示的解密层和用于增强护栏对抗基于模板的攻击的有效性的低等级适应机制。对超过22，000个提示的经验评估表明，与LlamaGuard和其他两个运行时护栏相比，DecipherGuard将RSC提高了36%至65%，将总体护栏性能（OGP）提高了20%至50%。这些结果凸显了DecipherGuard在保护LLM支持的软件系统在运行时免受越狱攻击方面的有效性。



## **15. AdaptiveGuard: Towards Adaptive Runtime Safety for LLM-Powered Software**

AdaptiveGuard：实现LLM支持的软件的自适应工作空间安全 cs.CR

Accepted to the ASE 2025 International Conference on Automated  Software Engineering, Industry Showcase Track

**SubmitDate**: 2025-09-21    [abs](http://arxiv.org/abs/2509.16861v1) [paper-pdf](http://arxiv.org/pdf/2509.16861v1)

**Authors**: Rui Yang, Michael Fu, Chakkrit Tantithamthavorn, Chetan Arora, Gunel Gulmammadova, Joey Chua

**Abstract**: Guardrails are critical for the safe deployment of Large Language Models (LLMs)-powered software. Unlike traditional rule-based systems with limited, predefined input-output spaces that inherently constrain unsafe behavior, LLMs enable open-ended, intelligent interactions--opening the door to jailbreak attacks through user inputs. Guardrails serve as a protective layer, filtering unsafe prompts before they reach the LLM. However, prior research shows that jailbreak attacks can still succeed over 70% of the time, even against advanced models like GPT-4o. While guardrails such as LlamaGuard report up to 95% accuracy, our preliminary analysis shows their performance can drop sharply--to as low as 12%--when confronted with unseen attacks. This highlights a growing software engineering challenge: how to build a post-deployment guardrail that adapts dynamically to emerging threats? To address this, we propose AdaptiveGuard, an adaptive guardrail that detects novel jailbreak attacks as out-of-distribution (OOD) inputs and learns to defend against them through a continual learning framework. Through empirical evaluation, AdaptiveGuard achieves 96% OOD detection accuracy, adapts to new attacks in just two update steps, and retains over 85% F1-score on in-distribution data post-adaptation, outperforming other baselines. These results demonstrate that AdaptiveGuard is a guardrail capable of evolving in response to emerging jailbreak strategies post deployment. We release our AdaptiveGuard and studied datasets at https://github.com/awsm-research/AdaptiveGuard to support further research.

摘要: 护栏对于安全部署大型语言模型（LLM）驱动的软件至关重要。与传统的基于规则的系统不同，LLM具有有限的、预定义的输入输出空间，本质上限制了不安全的行为，LLM可以实现开放式的智能交互--通过用户输入打开越狱攻击的大门。护栏充当保护层，在不安全的提示到达LLM之前对其进行过滤。然而，之前的研究表明，即使针对GPT-4 o等高级型号，越狱攻击仍然可以在70%以上的情况下成功。虽然LlamaGuard等护栏报告的准确率高达95%，但我们的初步分析显示，当面临不可见的攻击时，它们的性能可能会急剧下降-低至12%。这凸显了一个日益增长的软件工程挑战：如何构建一个动态适应新出现的威胁的部署后护栏？为了解决这个问题，我们提出了AdaptiveGuard，这是一种自适应护栏，可以将新的越狱攻击检测为分发外（OOD）输入，并通过持续学习框架来学习防御它们。通过实证评估，AdaptiveGuard实现了96%的OOD检测准确率，仅需两个更新步骤即可适应新攻击，并在适应后的分布数据上保持超过85%的F1分数，优于其他基线。这些结果表明，AdaptiveGuard是一种能够在部署后响应新兴越狱策略而不断发展的护栏。我们在https://github.com/awsm-research/AdaptiveGuard上发布了AdaptiveGuard并研究了数据集，以支持进一步的研究。



## **16. Design and Development of an Intelligent LLM-based LDAP Honeypot**

基于LLM的智能LDAP蜜罐的设计与开发 cs.CR

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2509.16682v1) [paper-pdf](http://arxiv.org/pdf/2509.16682v1)

**Authors**: Javier Jiménez-Román, Florina Almenares-Mendoza, Alfonso Sánchez-Macián

**Abstract**: Cybersecurity threats continue to increase, with a growing number of previously unknown attacks each year targeting both large corporations and smaller entities. This scenario demands the implementation of advanced security measures, not only to mitigate damage but also to anticipate emerging attack trends. In this context, deception tools have become a key strategy, enabling the detection, deterrence, and deception of potential attackers while facilitating the collection of information about their tactics and methods. Among these tools, honeypots have proven their value, although they have traditionally been limited by rigidity and configuration complexity, hindering their adaptability to dynamic scenarios. The rise of artificial intelligence, and particularly general-purpose Large Language Models (LLMs), is driving the development of new deception solutions capable of offering greater adaptability and ease of use. This work proposes the design and implementation of an LLM-based honeypot to simulate an LDAP server, a critical protocol present in most organizations due to its central role in identity and access management. The proposed solution aims to provide a flexible and realistic tool capable of convincingly interacting with attackers, thereby contributing to early detection and threat analysis while enhancing the defensive capabilities of infrastructures against intrusions targeting this service.

摘要: 网络安全威胁持续增加，每年针对大公司和小型实体的先前未知的攻击数量不断增加。这种情况需要实施先进的安全措施，不仅要减轻损害，还要预测新出现的攻击趋势。在这种情况下，欺骗工具已成为一种关键策略，能够检测、威慑和欺骗潜在攻击者，同时促进收集有关其策略和方法的信息。在这些工具中，蜜罐已经证明了自己的价值，尽管它们传统上受到刚性和配置复杂性的限制，阻碍了它们对动态场景的适应性。人工智能，特别是通用大型语言模型（LLM）的兴起，正在推动能够提供更大适应性和易用性的新型欺骗解决方案的开发。这项工作提出了设计和实现基于LLM的蜜罐来模拟PDA服务器，这是大多数组织中存在的关键协议，因为它在身份和访问管理中发挥了核心作用。拟议的解决方案旨在提供一种灵活且现实的工具，能够令人信服地与攻击者互动，从而有助于早期检测和威胁分析，同时增强基础设施针对该服务的入侵的防御能力。



## **17. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

糖衣毒药：良性一代解锁法学硕士越狱 cs.CR

Accepted by EMNLP2025

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2504.05652v3) [paper-pdf](http://arxiv.org/pdf/2504.05652v3)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Hao Zhang, Jia-Chen Zhang, Zheng Zhou

**Abstract**: With the increasingly deep integration of large language models (LLMs) across diverse domains, the effectiveness of their safety mechanisms is encountering severe challenges. Currently, jailbreak attacks based on prompt engineering have become a major safety threat. However, existing methods primarily rely on black-box manipulation of prompt templates, resulting in poor interpretability and limited generalization. To break through the bottleneck, this study first introduces the concept of Defense Threshold Decay (DTD), revealing the potential safety impact caused by LLMs' benign generation: as benign content generation in LLMs increases, the model's focus on input instructions progressively diminishes. Building on this insight, we propose the Sugar-Coated Poison (SCP) attack paradigm, which uses a "semantic reversal" strategy to craft benign inputs that are opposite in meaning to malicious intent. This strategy induces the models to generate extensive benign content, thereby enabling adversarial reasoning to bypass safety mechanisms. Experiments show that SCP outperforms existing baselines. Remarkably, it achieves an average attack success rate of 87.23% across six LLMs. For defense, we propose Part-of-Speech Defense (POSD), leveraging verb-noun dependencies for syntactic analysis to enhance safety of LLMs while preserving their generalization ability.

摘要: 随着大型语言模型（LLM）跨不同领域的日益深入集成，其安全机制的有效性面临严峻挑战。目前，基于即时工程的越狱攻击已成为重大安全威胁。然而，现有的方法主要依赖于提示模板的黑匣子操作，导致可解释性较差且概括性有限。为了突破瓶颈，本研究首先引入了防御阈值衰变（DART）的概念，揭示了LLM良性生成对安全的潜在影响：随着LLM良性内容生成的增加，模型对输入指令的关注逐渐减少。基于这一见解，我们提出了糖衣毒药（SCP）攻击范式，该范式使用“语义逆转”策略来制造与恶意意图含义相反的良性输入。该策略促使模型生成广泛的良性内容，从而使对抗推理能够绕过安全机制。实验表明SCP优于现有基线。值得注意的是，它在六个LLM中的平均攻击成功率为87.23%。对于防御，我们提出了词性防御（POSD），利用动词-名词依赖进行语法分析，以增强LLM的安全性，同时保留其概括能力。



## **18. FC-Attack: Jailbreaking Multimodal Large Language Models via Auto-Generated Flowcharts**

FC攻击：通过自动生成流程图破解多模式大型语言模型 cs.CV

Accepted to Findings of EMNLP 2025

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2502.21059v3) [paper-pdf](http://arxiv.org/pdf/2502.21059v3)

**Authors**: Ziyi Zhang, Zhen Sun, Zongmin Zhang, Jihui Guo, Xinlei He

**Abstract**: Multimodal Large Language Models (MLLMs) have become powerful and widely adopted in some practical applications. However, recent research has revealed their vulnerability to multimodal jailbreak attacks, whereby the model can be induced to generate harmful content, leading to safety risks. Although most MLLMs have undergone safety alignment, recent research shows that the visual modality is still vulnerable to jailbreak attacks. In our work, we discover that by using flowcharts with partially harmful information, MLLMs can be induced to provide additional harmful details. Based on this, we propose a jailbreak attack method based on auto-generated flowcharts, FC-Attack. Specifically, FC-Attack first fine-tunes a pre-trained LLM to create a step-description generator based on benign datasets. The generator is then used to produce step descriptions corresponding to a harmful query, which are transformed into flowcharts in 3 different shapes (vertical, horizontal, and S-shaped) as visual prompts. These flowcharts are then combined with a benign textual prompt to execute the jailbreak attack on MLLMs. Our evaluations on Advbench show that FC-Attack attains an attack success rate of up to 96% via images and up to 78% via videos across multiple MLLMs. Additionally, we investigate factors affecting the attack performance, including the number of steps and the font styles in the flowcharts. We also find that FC-Attack can improve the jailbreak performance from 4% to 28% in Claude-3.5 by changing the font style. To mitigate the attack, we explore several defenses and find that AdaShield can largely reduce the jailbreak performance but with the cost of utility drop.

摘要: 多模式大型语言模型（MLLM）已变得强大并在一些实际应用中广泛采用。然而，最近的研究揭示了它们对多模式越狱攻击的脆弱性，从而可以诱导模型生成有害内容，从而导致安全风险。尽管大多数MLLM都经历了安全调整，但最近的研究表明，视觉模式仍然容易受到越狱攻击。在我们的工作中，我们发现通过使用包含部分有害信息的流程图，可能会诱导MLLM提供额外的有害细节。基于此，我们提出了一种基于自动生成流程图的越狱攻击方法FC-Attack。具体来说，FC-Attack首先微调预训练的LLM，以基于良性数据集创建步骤描述生成器。然后使用生成器生成与有害查询对应的步骤描述，并将其转换为3种不同形状（垂直、水平和S形）的流程图作为视觉提示。然后，这些流程图与良性文本提示相结合，对MLLM执行越狱攻击。我们对Advbridge的评估显示，FC-Attack在多个MLLM中通过图像获得的攻击成功率高达96%，通过视频获得的攻击成功率高达78%。此外，我们还调查了影响攻击性能的因素，包括步骤数和流程图中的字体样式。我们还发现，通过改变字体风格，FC-Attack可以将Claude-3.5中的越狱性能从4%提高到28%。为了减轻攻击，我们探索了几种防御措施，发现AdaShield可以大大降低越狱性能，但公用事业成本会下降。



## **19. Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking**

推理防御：安全意识推理可以保护大型语言模型免受越狱 cs.CL

EMNLP 2025

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2502.12970v3) [paper-pdf](http://arxiv.org/pdf/2502.12970v3)

**Authors**: Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, Lei Sha

**Abstract**: Large Reasoning Models (LRMs) have recently demonstrated impressive performances across diverse domains. However, how the safety of Large Language Models (LLMs) benefits from enhanced reasoning capabilities against jailbreak queries remains unexplored. To bridge this gap, in this paper, we propose Reasoning-to-Defend (R2D), a novel training paradigm that integrates a safety-aware reasoning mechanism into LLMs' generation process. This enables self-evaluation at each step of the reasoning process, forming safety pivot tokens as indicators of the safety status of responses. Furthermore, in order to improve the accuracy of predicting pivot tokens, we propose Contrastive Pivot Optimization (CPO), which enhances the model's perception of the safety status of given dialogues. LLMs dynamically adjust their response strategies during reasoning, significantly enhancing their safety capabilities defending jailbreak attacks. Extensive experiments demonstrate that R2D effectively mitigates various attacks and improves overall safety, while maintaining the original performances. This highlights the substantial potential of safety-aware reasoning in improving robustness of LRMs and LLMs against various jailbreaks.

摘要: 大型推理模型（LRM）最近在不同领域展示了令人印象深刻的性能。然而，大型语言模型（LLM）的安全性如何从针对越狱查询的增强推理能力中受益仍有待探索。为了弥合这一差距，在本文中，我们提出了推理防御（R2 D），这是一种新型训练范式，将安全感知推理机制集成到LLM的生成过程中。这使得推理过程的每个步骤都能够进行自我评估，形成安全支点令牌作为响应安全状态的指标。此外，为了提高预测枢纽令牌的准确性，我们提出了对比枢纽优化（CPO），它增强了模型对给定对话安全状态的感知。LLM在推理过程中动态调整响应策略，显着增强了防御越狱攻击的安全能力。大量实验表明，R2D有效地缓解了各种攻击并提高了整体安全性，同时保持了原始性能。这凸显了安全意识推理在提高LRM和LLM针对各种越狱的稳健性方面的巨大潜力。



## **20. MIST: Jailbreaking Black-box Large Language Models via Iterative Semantic Tuning**

MIST：通过迭代语义调优破解黑匣子大型语言模型 cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2506.16792v3) [paper-pdf](http://arxiv.org/pdf/2506.16792v3)

**Authors**: Muyang Zheng, Yuanzhi Yao, Changting Lin, Caihong Kai, Yanxiang Chen, Zhiquan Liu

**Abstract**: Despite efforts to align large language models (LLMs) with societal and moral values, these models remain susceptible to jailbreak attacks -- methods designed to elicit harmful responses. Jailbreaking black-box LLMs is considered challenging due to the discrete nature of token inputs, restricted access to the target LLM, and limited query budget. To address the issues above, we propose an effective method for jailbreaking black-box large language Models via Iterative Semantic Tuning, named MIST. MIST enables attackers to iteratively refine prompts that preserve the original semantic intent while inducing harmful content. Specifically, to balance semantic similarity with computational efficiency, MIST incorporates two key strategies: sequential synonym search, and its advanced version -- order-determining optimization. We conduct extensive experiments on two datasets using two open-source and four closed-source models. Results show that MIST achieves competitive attack success rate, relatively low query count, and fair transferability, outperforming or matching state-of-the-art jailbreak methods. Additionally, we conduct analysis on computational efficiency to validate the practical viability of MIST.

摘要: 尽管人们努力将大型语言模型（LLM）与社会和道德价值观保持一致，但这些模型仍然容易受到越狱攻击--这些攻击旨在引发有害反应的方法。由于令牌输入的离散性、对目标LLM的访问受限以及查询预算有限，越狱黑匣子LLM被认为具有挑战性。为了解决上述问题，我们提出了一种通过迭代语义调优破解黑匣子大型语言模型的有效方法，名为MIST。MIST使攻击者能够迭代地改进提示，以保留原始语义意图，同时诱导有害内容。具体来说，为了平衡语义相似性与计算效率，MIST结合了两个关键策略：顺序同义词搜索及其高级版本--顺序确定优化。我们使用两个开源模型和四个开源模型对两个数据集进行了广泛的实验。结果表明，MIST实现了有竞争力的攻击成功率、相对较低的查询计数和公平的可移植性，优于或匹配最先进的越狱方法。此外，我们还对计算效率进行分析，以验证MIST的实际可行性。



## **21. Can an Individual Manipulate the Collective Decisions of Multi-Agents?**

个人可以操纵多主体的集体决策吗？ cs.CL

**SubmitDate**: 2025-09-20    [abs](http://arxiv.org/abs/2509.16494v1) [paper-pdf](http://arxiv.org/pdf/2509.16494v1)

**Authors**: Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, Jindong Gu

**Abstract**: Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.

摘要: 个体大型语言模型（LLM）已在医疗保健和法律等各个领域展现出强大的能力。最近的研究还表明，协调的多智能体系统通过协作表现出增强的决策和推理能力。然而，由于单个LLM的脆弱性以及访问多代理系统中所有代理的困难，出现了一个关键问题：如果攻击者只知道一个代理，他们还能生成能够误导集体决策的对抗样本吗？为了探索这个问题，我们将其描述为一个信息不完整的游戏，其中攻击者只知道一个目标代理，并且缺乏对系统中其他代理的了解。通过这个公式，我们提出了M-Spoiler，这是一个模拟多智能体系统内的智能体交互以生成对抗样本的框架。然后使用这些样本来操纵目标系统中的目标代理，误导系统的协作决策过程。更具体地说，M-Spoiler引入了一种顽固代理，它通过模拟目标系统中代理的潜在顽固反应来积极帮助优化对抗样本。这增强了生成的对抗样本误导系统的有效性。通过针对各种任务的广泛实验，我们的研究结果证实了多代理系统中单个代理的知识所带来的风险，并证明了我们框架的有效性。我们还探索了几种防御机制，表明我们提出的攻击框架仍然比基线更有效，强调了进一步研究防御策略的必要性。



## **22. From Capabilities to Performance: Evaluating Key Functional Properties of LLM Architectures in Penetration Testing**

从能力到性能：在渗透测试中评估LLM架构的关键功能属性 cs.AI

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.14289v2) [paper-pdf](http://arxiv.org/pdf/2509.14289v2)

**Authors**: Lanxiao Huang, Daksh Dave, Ming Jin, Tyler Cody, Peter Beling

**Abstract**: Large language models (LLMs) are increasingly used to automate or augment penetration testing, but their effectiveness and reliability across attack phases remain unclear. We present a comprehensive evaluation of multiple LLM-based agents, from single-agent to modular designs, across realistic penetration testing scenarios, measuring empirical performance and recurring failure patterns. We also isolate the impact of five core functional capabilities via targeted augmentations: Global Context Memory (GCM), Inter-Agent Messaging (IAM), Context-Conditioned Invocation (CCI), Adaptive Planning (AP), and Real-Time Monitoring (RTM). These interventions support, respectively: (i) context coherence and retention, (ii) inter-component coordination and state management, (iii) tool use accuracy and selective execution, (iv) multi-step strategic planning, error detection, and recovery, and (v) real-time dynamic responsiveness. Our results show that while some architectures natively exhibit subsets of these properties, targeted augmentations substantially improve modular agent performance, especially in complex, multi-step, and real-time penetration testing tasks.

摘要: 大型语言模型（LLM）越来越多地用于自动化或增强渗透测试，但它们在攻击阶段的有效性和可靠性仍不清楚。我们在现实的渗透测试场景中对多个基于LLM的代理（从单代理到模块化设计）进行了全面评估，测量经验性能和反复出现的故障模式。我们还通过有针对性的增强来隔离五种核心功能能力的影响：全球上下文记忆（GCM）、代理间消息传递（ILM）、上下文条件调用（CI）、自适应规划（AP）和实时监控（RTI）。这些干预措施分别支持：（i）上下文一致性和保留，（ii）组件间协调和状态管理，（iii）工具使用准确性和选择性执行，（iv）多步骤战略规划、错误检测和恢复，以及（v）实时动态响应能力。我们的结果表明，虽然一些架构本身表现出这些属性的子集，但有针对性的增强可以大大提高模块化代理的性能，特别是在复杂、多步骤和实时渗透测试任务中。



## **23. Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs**

目标对齐：提取对齐的LLM的安全分类器 cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2501.16534v2) [paper-pdf](http://arxiv.org/pdf/2501.16534v2)

**Authors**: Jean-Charles Noirot Ferrand, Yohan Beugin, Eric Pauley, Ryan Sheatsley, Patrick McDaniel

**Abstract**: Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we introduce and evaluate a new technique for jailbreak attacks. We observe that alignment embeds a safety classifier in the LLM responsible for deciding between refusal and compliance, and seek to extract an approximation of this classifier: a surrogate classifier. To this end, we build candidate classifiers from subsets of the LLM. We first evaluate the degree to which candidate classifiers approximate the LLM's safety classifier in benign and adversarial settings. Then, we attack the candidates and measure how well the resulting adversarial inputs transfer to the LLM. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find that attacks mounted on the surrogate classifiers can be transferred to the LLM with high success. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70% with half the memory footprint and runtime -- a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is an effective and efficient means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks.

摘要: 大型语言模型（LLM）中的对齐用于强制执行安全等准则。然而，面对越狱攻击，调整失败了，这些攻击修改了输入以引发不安全的输出。本文介绍并评估了一种新的越狱攻击技术。我们观察到，对齐在LLM中嵌入了一个安全分类器，负责在拒绝和合规之间做出决定，并试图提取该分类器的近似值：代理分类器。为此，我们从LLM的子集中构建候选分类器。我们首先评估候选分类器在良性和对抗环境中接近LLM安全分类器的程度。然后，我们攻击候选人并衡量由此产生的对抗输入转移到LLM的程度。我们的评估表明，最好的候选人只需使用20%的模型架构即可实现准确的一致性（F1评分高于80%）。此外，我们发现，安装在代理分类器上的攻击可以转移到LLM，具有很高的成功率。例如，仅使用50%的Llama 2模型的代理实现了70%的攻击成功率（ASR），而内存占用和运行时间只有一半-与直接攻击LLM相比有了实质性的改进，我们只观察到22%的ASR。这些结果表明，提取代理分类器是一种有效和高效的手段建模（并在其中解决）的漏洞对齐模型越狱攻击。



## **24. On the Security of Tool-Invocation Prompts for LLM-Based Agentic Systems: An Empirical Risk Assessment**

基于LLM的统计系统工具调用预算的安全性：经验风险评估 cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.05755v4) [paper-pdf](http://arxiv.org/pdf/2509.05755v4)

**Authors**: Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Yu Liu, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.

摘要: 基于法学硕士的代理系统利用大型语言模型来处理用户查询、做出决策并执行外部工具，以执行跨聊天机器人、客户服务和软件工程等领域的复杂任务。这些系统的一个关键组件是工具调用提示（TIP），它定义了工具交互协议并指导LLM确保工具使用的安全性和正确性。尽管TIP的安全性很重要，但在很大程度上被忽视了。这项工作调查了与TIP相关的安全风险，揭示了Cursor、Claude Code等基于LLM的主要系统容易受到远程代码执行（RCE）和拒绝服务（NOS）等攻击。通过系统性TIP利用工作流程（TEW），我们通过操纵工具调用演示了外部工具行为劫持。我们还提出了防御机制来增强基于LLM的代理系统中的TIP安全性。



## **25. SABER: Uncovering Vulnerabilities in Safety Alignment via Cross-Layer Residual Connection**

SABER：通过跨层剩余连接揭示安全对齐中的漏洞 cs.LG

Accepted in EMNLP'25 Main

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.16060v1) [paper-pdf](http://arxiv.org/pdf/2509.16060v1)

**Authors**: Maithili Joshi, Palash Nandi, Tanmoy Chakraborty

**Abstract**: Large Language Models (LLMs) with safe-alignment training are powerful instruments with robust language comprehension capabilities. These models typically undergo meticulous alignment procedures involving human feedback to ensure the acceptance of safe inputs while rejecting harmful or unsafe ones. However, despite their massive scale and alignment efforts, LLMs remain vulnerable to jailbreak attacks, where malicious users manipulate the model to produce harmful outputs that it was explicitly trained to avoid. In this study, we find that the safety mechanisms in LLMs are predominantly embedded in the middle-to-late layers. Building on this insight, we introduce a novel white-box jailbreak method, SABER (Safety Alignment Bypass via Extra Residuals), which connects two intermediate layers $s$ and $e$ such that $s < e$, through a residual connection. Our approach achieves a 51% improvement over the best-performing baseline on the HarmBench test set. Furthermore, SABER induces only a marginal shift in perplexity when evaluated on the HarmBench validation set. The source code is publicly available at https://github.com/PalGitts/SABER.

摘要: 具有安全对齐培训的大型语言模型（LLM）是具有强大语言理解能力的强大工具。这些模型通常经过涉及人类反馈的细致的调整程序，以确保接受安全输入，同时拒绝有害或不安全的输入。然而，尽管LLM在规模和协调方面做出了巨大的努力，但它们仍然容易受到越狱攻击，恶意用户操纵模型以产生有害输出，而这些输出经过明确训练可以避免。在这项研究中，我们发现LLM中的安全机制主要嵌入在中后期。基于这一见解，我们引入了一种新颖的白盒越狱方法SABER（通过额外剩余量进行安全对齐旁路），它通过剩余连接连接两个中间层$s$和$e$，使得$s < e$。我们的方法比HarmBench测试集的最佳性能基线提高了51%。此外，当在HarmBench验证集中进行评估时，SABER只会导致困惑的边际变化。源代码可在https://github.com/PalGitts/SABER上公开获取。



## **26. Tag&Tab: Pretraining Data Detection in Large Language Models Using Keyword-Based Membership Inference Attack**

标记& Tab：使用基于关键字的成员推断攻击在大型语言模型中进行预训练数据检测 cs.CR

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2501.08454v2) [paper-pdf](http://arxiv.org/pdf/2501.08454v2)

**Authors**: Sagiv Antebi, Edan Habler, Asaf Shabtai, Yuval Elovici

**Abstract**: Large language models (LLMs) have become essential tools for digital task assistance. Their training relies heavily on the collection of vast amounts of data, which may include copyright-protected or sensitive information. Recent studies on detecting pretraining data in LLMs have primarily focused on sentence- or paragraph-level membership inference attacks (MIAs), usually involving probability analysis of the target model's predicted tokens. However, these methods often exhibit poor accuracy, failing to account for the semantic importance of textual content and word significance. To address these shortcomings, we propose Tag&Tab, a novel approach for detecting data used in LLM pretraining. Our method leverages established natural language processing (NLP) techniques to tag keywords in the input text, a process we term Tagging. Then, the LLM is used to obtain probabilities for these keywords and calculate their average log-likelihood to determine input text membership, a process we refer to as Tabbing. Our experiments on four benchmark datasets (BookMIA, MIMIR, PatentMIA, and the Pile) and several open-source LLMs of varying sizes demonstrate an average increase in AUC scores ranging from 5.3% to 17.6% over state-of-the-art methods. Tag&Tab not only sets a new standard for data leakage detection in LLMs, but its outstanding performance is a testament to the importance of words in MIAs on LLMs.

摘要: 大型语言模型（LLM）已成为数字任务辅助的重要工具。他们的培训严重依赖于大量数据的收集，其中可能包括受版权保护的或敏感信息。最近关于在LLM中检测预训练数据的研究主要集中在句子或段落级成员资格推理攻击（MIA），通常涉及目标模型预测令牌的概率分析。然而，这些方法的准确性通常很差，未能考虑文本内容和单词重要性的语义重要性。为了解决这些缺点，我们提出了TAG & Tab，这是一种检测LLM预训练中使用的数据的新型方法。我们的方法利用既定的自然语言处理（NLP）技术来标记输入文本中的关键词，我们将此过程称为“标记”。然后，使用LLM来获取这些关键词的概率，并计算其平均日志似然性以确定输入文本成员资格，我们将此过程称为Tabbing。我们对四个基准数据集（BookMIA、MIIR、PatentMIA和Pile）和几个不同规模的开源LLM进行的实验表明，与最先进的方法相比，AUT评分平均增加了5.3%至17.6%。TAG & Tab不仅为LLM中的数据泄露检测设定了新标准，而且其出色的性能也证明了LLM上MIA中单词的重要性。



## **27. Can LLMs Judge Debates? Evaluating Non-Linear Reasoning via Argumentation Theory Semantics**

LLM可以判断辩论吗？用论证理论语义评价非线性推理 cs.CL

Accepted to EMNLP 2025 Findings

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15739v1) [paper-pdf](http://arxiv.org/pdf/2509.15739v1)

**Authors**: Reza Sanayei, Srdjan Vesic, Eduardo Blanco, Mihai Surdeanu

**Abstract**: Large Language Models (LLMs) excel at linear reasoning tasks but remain underexplored on non-linear structures such as those found in natural debates, which are best expressed as argument graphs. We evaluate whether LLMs can approximate structured reasoning from Computational Argumentation Theory (CAT). Specifically, we use Quantitative Argumentation Debate (QuAD) semantics, which assigns acceptability scores to arguments based on their attack and support relations. Given only dialogue-formatted debates from two NoDE datasets, models are prompted to rank arguments without access to the underlying graph. We test several LLMs under advanced instruction strategies, including Chain-of-Thought and In-Context Learning. While models show moderate alignment with QuAD rankings, performance degrades with longer inputs or disrupted discourse flow. Advanced prompting helps mitigate these effects by reducing biases related to argument length and position. Our findings highlight both the promise and limitations of LLMs in modeling formal argumentation semantics and motivate future work on graph-aware reasoning.

摘要: 大型语言模型（LLM）擅长线性推理任务，但对非线性结构（例如自然辩论中发现的结构）的探索不足，这些结构最好用参数图来表达。我们评估LLM是否可以从计算论证理论（CAT）中逼近结构化推理。具体来说，我们使用量化论证辩论（QuAD）语义，该语义根据论点的攻击和支持关系为论点分配可接受性分数。仅考虑来自两个NoDE数据集的对话框格式辩论，模型会在不访问底层图表的情况下对参数进行排名。我们在先进的教学策略下测试了多个LLM，包括思想链和上下文学习。虽然模型显示出与QuAD排名适度一致，但随着输入时间的延长或话语流的中断，性能会下降。高级提示通过减少与论点长度和立场相关的偏见来帮助减轻这些影响。我们的研究结果强调了LLM在建模形式论证语义方面的前景和局限性，并激励未来在图形感知推理方面的工作。



## **28. AdaSteer: Your Aligned LLM is Inherently an Adaptive Jailbreak Defender**

AdaSteer：您的LLM本质上是一个适应性越狱捍卫者 cs.CR

19 pages, 6 figures, 10 tables

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2504.09466v2) [paper-pdf](http://arxiv.org/pdf/2504.09466v2)

**Authors**: Weixiang Zhao, Jiahe Guo, Yulin Hu, Yang Deng, An Zhang, Xingyu Sui, Xinyang Han, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu

**Abstract**: Despite extensive efforts in safety alignment, large language models (LLMs) remain vulnerable to jailbreak attacks. Activation steering offers a training-free defense method but relies on fixed steering coefficients, resulting in suboptimal protection and increased false rejections of benign inputs. To address this, we propose AdaSteer, an adaptive activation steering method that dynamically adjusts model behavior based on input characteristics. We identify two key properties: Rejection Law (R-Law), which shows that stronger steering is needed for jailbreak inputs opposing the rejection direction, and Harmfulness Law (H-Law), which differentiates adversarial and benign inputs. AdaSteer steers input representations along both the Rejection Direction (RD) and Harmfulness Direction (HD), with adaptive coefficients learned via logistic regression, ensuring robust jailbreak defense while preserving benign input handling. Experiments on LLaMA-3.1, Gemma-2, and Qwen2.5 show that AdaSteer outperforms baseline methods across multiple jailbreak attacks with minimal impact on utility. Our results highlight the potential of interpretable model internals for real-time, flexible safety enforcement in LLMs.

摘要: 尽管在安全调整方面做出了广泛的努力，但大型语言模型（LLM）仍然容易受到越狱攻击。激活转向提供了一种无需训练的防御方法，但依赖于固定的转向系数，从而导致次优保护和良性输入的错误拒绝增加。为了解决这个问题，我们提出了AdaSteer，这是一种自适应激活引导方法，可以根据输入特征动态调整模型行为。我们确定了两个关键属性：拒绝定律（R-Law），它表明与拒绝方向相反的越狱输入需要更强的引导，以及有害定律（H-Law），它区分对抗性和良性输入。AdaSteer沿着拒绝方向（RD）和有害方向（HD）引导输入表示，通过逻辑回归学习自适应系数，确保强大的越狱防御，同时保持良性的输入处理。在LLaMA-3.1、Gemma-2和Qwen2.5上的实验表明，AdaSteer在多种越狱攻击中的性能优于基线方法，对效用的影响最小。我们的研究结果突出了可解释的模型内部实时，灵活的安全执法LLM的潜力。



## **29. DNA-DetectLLM: Unveiling AI-Generated Text via a DNA-Inspired Mutation-Repair Paradigm**

DNA-DetectLLM：通过DNA启发的突变修复范式揭示人工智能生成的文本 cs.CL

NeurIPS 2025 Spotlight

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2509.15550v1) [paper-pdf](http://arxiv.org/pdf/2509.15550v1)

**Authors**: Xiaowei Zhu, Yubing Ren, Fang Fang, Qingfeng Tan, Shi Wang, Yanan Cao

**Abstract**: The rapid advancement of large language models (LLMs) has blurred the line between AI-generated and human-written text. This progress brings societal risks such as misinformation, authorship ambiguity, and intellectual property concerns, highlighting the urgent need for reliable AI-generated text detection methods. However, recent advances in generative language modeling have resulted in significant overlap between the feature distributions of human-written and AI-generated text, blurring classification boundaries and making accurate detection increasingly challenging. To address the above challenges, we propose a DNA-inspired perspective, leveraging a repair-based process to directly and interpretably capture the intrinsic differences between human-written and AI-generated text. Building on this perspective, we introduce DNA-DetectLLM, a zero-shot detection method for distinguishing AI-generated and human-written text. The method constructs an ideal AI-generated sequence for each input, iteratively repairs non-optimal tokens, and quantifies the cumulative repair effort as an interpretable detection signal. Empirical evaluations demonstrate that our method achieves state-of-the-art detection performance and exhibits strong robustness against various adversarial attacks and input lengths. Specifically, DNA-DetectLLM achieves relative improvements of 5.55% in AUROC and 2.08% in F1 score across multiple public benchmark datasets.

摘要: 大型语言模型（LLM）的快速发展模糊了人工智能生成的文本和人类编写的文本之间的界限。这一进展带来了错误信息、作者身份模糊和知识产权问题等社会风险，凸显了对可靠的人工智能生成文本检测方法的迫切需求。然而，生成式语言建模的最新进展导致人类书写文本和人工智能生成文本的特征分布之间存在显着重叠，模糊了分类边界，并使准确检测变得越来越具有挑战性。为了解决上述挑战，我们提出了一种受DNA启发的视角，利用基于修复的流程来直接且可解释地捕捉人类书写和人工智能生成的文本之间的内在差异。基于这一观点，我们引入了DNA-DetectLLM，这是一种用于区分人工智能生成文本和人类编写文本的零镜头检测方法。该方法为每个输入构建理想的人工智能生成序列，迭代地修复非最优令牌，并将累积修复工作量化为可解释的检测信号。经验评估表明，我们的方法实现了最先进的检测性能，并对各种对抗攻击和输入长度表现出强大的鲁棒性。具体来说，在多个公共基准数据集中，DNA-DetectLLM在AUROC和F1评分上相对提高了5.55%，在F1评分上相对提高了2.08%。



## **30. SecReEvalBench: A Multi-turned Security Resilience Evaluation Benchmark for Large Language Models**

SecReEvalBench：大型语言模型的多角度安全弹性评估基准 cs.CR

Major rework on the paper that changes the title, content,  experiments, story, and etc. All authors agree to withdraw

**SubmitDate**: 2025-09-19    [abs](http://arxiv.org/abs/2505.07584v3) [paper-pdf](http://arxiv.org/pdf/2505.07584v3)

**Authors**: Huining Cui, Wei Liu

**Abstract**: The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cybersecurity attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turn attacks. To address this gap, we present SecReEvalBench, the Security Resilience Evaluation Benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain-Based Attack Resilience Score and Chain-Based Attack Rejection Time Score. Moreover, SecReEvalBench employs six questioning sequences for model assessment: one-off attack, successive attack, successive reverse attack, alternative attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels. In addition, we introduce a dataset customized for the benchmark, which incorporates both neutral and malicious prompts, categorised across seven security domains and sixteen attack techniques. In applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats. The SecReEvalBench dataset is publicly available at https://kaggle.com/datasets/5a7ee22cf9dab6c93b55a73f630f6c9b42e936351b0ae98fbae6ddaca7fe248d, which provides a groundwork for advancing research in large language model security.

摘要: 大型语言模型在安全敏感领域的部署越来越多，需要严格评估它们对抗基于预算的敌对攻击的弹性。虽然之前的基准侧重于有限且预定义的攻击域（例如网络安全攻击）的安全评估，但它们通常缺乏对意图驱动的对抗提示的全面评估以及对现实生活中基于情景的多回合攻击的考虑。为了解决这一差距，我们提出了SecReEvalBench，安全韧性评估基准，它定义了四个新颖的指标：即时攻击韧性分数、即时攻击拒绝逻辑分数、基于链的攻击韧性分数和基于链的攻击拒绝时间分数。此外，SecReEvalBench采用六个提问序列进行模型评估：一次性攻击、连续攻击、连续反向攻击、替代攻击、威胁级别不断上升的顺序上升攻击和威胁级别不断下降的顺序下降攻击。此外，我们还引入了一个为基准定制的数据集，其中包含中性和恶意提示，分为七个安全域和十六种攻击技术。在应用该基准时，我们系统地评估了五个最先进的开放加权大型语言模型：Llama 3.1、Gemma 2、Mistral v0.3、DeepSeek-R1和Qwen 3。我们的研究结果为现代大型语言模型在防御不断变化的对抗威胁方面的优势和弱点提供了重要的见解。SecReEvalBench数据集可在https：//kaggle.com/guardets/5a7ee22CF9dab6c93b55a73f630f6c9 b42 e936351 b 0ae 98 fbae 6ddaca 7 fe 248 d上公开，为推进大型语言模型安全性研究提供了基础。



## **31. ORCA: Agentic Reasoning For Hallucination and Adversarial Robustness in Vision-Language Models**

ORCA：视觉语言模型中幻觉和对抗鲁棒性的抽象推理 cs.CV

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15435v1) [paper-pdf](http://arxiv.org/pdf/2509.15435v1)

**Authors**: Chung-En Johnny Yu, Hsuan-Chih, Chen, Brian Jalaian, Nathaniel D. Bastian

**Abstract**: Large Vision-Language Models (LVLMs) exhibit strong multimodal capabilities but remain vulnerable to hallucinations from intrinsic errors and adversarial attacks from external exploitations, limiting their reliability in real-world applications. We present ORCA, an agentic reasoning framework that improves the factual accuracy and adversarial robustness of pretrained LVLMs through test-time structured inference reasoning with a suite of small vision models (less than 3B parameters). ORCA operates via an Observe--Reason--Critique--Act loop, querying multiple visual tools with evidential questions, validating cross-model inconsistencies, and refining predictions iteratively without access to model internals or retraining. ORCA also stores intermediate reasoning traces, which supports auditable decision-making. Though designed primarily to mitigate object-level hallucinations, ORCA also exhibits emergent adversarial robustness without requiring adversarial training or defense mechanisms. We evaluate ORCA across three settings: (1) clean images on hallucination benchmarks, (2) adversarially perturbed images without defense, and (3) adversarially perturbed images with defense applied. On the POPE hallucination benchmark, ORCA improves standalone LVLM performance by +3.64\% to +40.67\% across different subsets. Under adversarial perturbations on POPE, ORCA achieves an average accuracy gain of +20.11\% across LVLMs. When combined with defense techniques on adversarially perturbed AMBER images, ORCA further improves standalone LVLM performance, with gains ranging from +1.20\% to +48.00\% across evaluation metrics. These results demonstrate that ORCA offers a promising path toward building more reliable and robust multimodal systems.

摘要: 大型视觉语言模型（LVLM）展现出强大的多模式能力，但仍然容易受到内在错误和外部利用的对抗攻击的幻觉，限制了它们在现实世界应用中的可靠性。我们提出了ORCA，这是一个代理推理框架，通过使用一套小视觉模型（小于3B参数）的测试时结构化推理来提高预训练LVLM的事实准确性和对抗鲁棒性。ORCA通过观察--原因--批评--行为循环运行，通过证据问题查询多个视觉工具，验证跨模型的不一致性，并在不访问模型内部或再培训的情况下迭代改进预测。ORCA还存储中间推理痕迹，支持可审计决策。尽管ORCA主要设计用于减轻对象级幻觉，但它也表现出紧急对抗鲁棒性，而不需要对抗训练或防御机制。我们在三种设置中评估ORCA：（1）幻觉基准上的干净图像，（2）在没有防御的情况下对抗性干扰图像，以及（3）应用防御的情况下对抗性干扰图像。在POPE幻觉基准测试中，ORCA在不同子集中将独立LVLM性能提高了+3.64\%至+40.67\%。在POPE的对抗性扰动下，ORCA在LVLM中实现了+20.11%的平均准确性提高。当与对抗干扰的AMBER图像上的防御技术相结合时，ORCA进一步提高了独立LVLM性能，评估指标的收益范围从+1.20\%到+48.00\%不等。这些结果表明，ORCA为构建更可靠、更稳健的多模式系统提供了一条有希望的途径。



## **32. Evil Vizier: Vulnerabilities of LLM-Integrated XR Systems**

Evil Vizier：LLM集成XR系统的漏洞 cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15213v1) [paper-pdf](http://arxiv.org/pdf/2509.15213v1)

**Authors**: Yicheng Zhang, Zijian Huang, Sophie Chen, Erfan Shayegani, Jiasi Chen, Nael Abu-Ghazaleh

**Abstract**: Extended reality (XR) applications increasingly integrate Large Language Models (LLMs) to enhance user experience, scene understanding, and even generate executable XR content, and are often called "AI glasses". Despite these potential benefits, the integrated XR-LLM pipeline makes XR applications vulnerable to new forms of attacks. In this paper, we analyze LLM-Integated XR systems in the literature and in practice and categorize them along different dimensions from a systems perspective. Building on this categorization, we identify a common threat model and demonstrate a series of proof-of-concept attacks on multiple XR platforms that employ various LLM models (Meta Quest 3, Meta Ray-Ban, Android, and Microsoft HoloLens 2 running Llama and GPT models). Although these platforms each implement LLM integration differently, they share vulnerabilities where an attacker can modify the public context surrounding a legitimate LLM query, resulting in erroneous visual or auditory feedback to users, thus compromising their safety or privacy, sowing confusion, or other harmful effects. To defend against these threats, we discuss mitigation strategies and best practices for developers, including an initial defense prototype, and call on the community to develop new protection mechanisms to mitigate these risks.

摘要: 延展实境（XR）应用程序越来越多地集成大型语言模型（LLM），以增强用户体验、场景理解，甚至生成可执行XR内容，通常被称为“AI眼镜”。尽管有这些潜在的好处，但集成的XR-LLM管道使XR应用程序容易受到新形式的攻击。在本文中，我们分析了LLM-Integated XR系统在文献和实践中，并从系统的角度沿着不同的维度对它们进行分类。在此分类的基础上，我们识别了常见的威胁模型，并在采用各种LLM模型（Meta Quest 3、Meta Ray-Ban、Android和运行Lama和GPT模型的Microsoft HoloLens 2）的多个XR平台上演示了一系列概念验证攻击。尽管这些平台各自以不同的方式实现LLM集成，但它们都有漏洞，攻击者可以修改围绕合法LLM查询的公共上下文，从而导致向用户提供错误的视觉或听觉反馈，从而损害他们的安全或隐私、散布混乱或其他有害影响。为了抵御这些威胁，我们讨论了开发人员的缓解策略和最佳实践，包括初始的防御原型，并呼吁社区开发新的保护机制来缓解这些风险。



## **33. Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction**

超越表面对齐：通过概率简化拒绝指示重建LLM安全机制 cs.CR

Accepted by EMNLP2025 Finding

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15202v1) [paper-pdf](http://arxiv.org/pdf/2509.15202v1)

**Authors**: Yuanbo Xie, Yingjie Zhang, Tianyun Liu, Duohe Ma, Tingwen Liu

**Abstract**: Jailbreak attacks pose persistent threats to large language models (LLMs). Current safety alignment methods have attempted to address these issues, but they experience two significant limitations: insufficient safety alignment depth and unrobust internal defense mechanisms. These limitations make them vulnerable to adversarial attacks such as prefilling and refusal direction manipulation. We introduce DeepRefusal, a robust safety alignment framework that overcomes these issues. DeepRefusal forces the model to dynamically rebuild its refusal mechanisms from jailbreak states. This is achieved by probabilistically ablating the refusal direction across layers and token depths during fine-tuning. Our method not only defends against prefilling and refusal direction attacks but also demonstrates strong resilience against other unseen jailbreak strategies. Extensive evaluations on four open-source LLM families and six representative attacks show that DeepRefusal reduces attack success rates by approximately 95%, while maintaining model capabilities with minimal performance degradation.

摘要: 越狱攻击对大型语言模型（LLM）构成持续威胁。当前的安全对齐方法试图解决这些问题，但它们遇到了两个重大局限性：安全对齐深度不足和内部防御机制不健全。这些限制使它们容易受到预填充和拒绝方向操纵等敌对攻击。我们引入DeepRefusal，这是一个强大的安全调整框架，可以克服这些问题。DeepRefusal迫使该模型动态重建其来自越狱国家的拒绝机制。这是通过在微调期间概率消除跨层和代币深度的拒绝方向来实现的。我们的方法不仅可以抵御预填充和拒绝方向攻击，而且还表现出对其他看不见的越狱策略的强大韧性。对四个开源LLM系列和六种代表性攻击的广泛评估表明，DeepRefusal将攻击成功率降低了约95%，同时以最小的性能下降保持模型能力。



## **34. QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA**

QA-LIGN：通过宪法分解的QA调整LLM cs.CL

Accepted to Findings of EMNLP 2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2506.08123v3) [paper-pdf](http://arxiv.org/pdf/2506.08123v3)

**Authors**: Jacob Dineen, Aswin RRV, Qin Liu, Zhikun Xu, Xiao Ye, Ming Shen, Zhaonan Li, Shijie Lu, Chitta Baral, Muhao Chen, Ben Zhou

**Abstract**: Alignment of large language models (LLMs) with principles like helpfulness, honesty, and harmlessness typically relies on scalar rewards that obscure which objectives drive the training signal. We introduce QA-LIGN, which decomposes monolithic rewards into interpretable principle-specific evaluations through structured natural language programs. Models learn through a draft, critique, and revise pipeline, where symbolic evaluation against the rubrics provides transparent feedback for both initial and revised responses during GRPO training. Applied to uncensored Llama-3.1-8B-Instruct, QA-LIGN reduces attack success rates by up to 68.7% while maintaining a 0.67% false refusal rate, achieving Pareto optimal safety-helpfulness performance and outperforming both DPO and GRPO with state-of-the-art reward models given equivalent training. These results demonstrate that making reward signals interpretable and modular improves alignment effectiveness, suggesting transparency enhances LLM safety.

摘要: 大型语言模型（LLM）与乐于助人、诚实和无害等原则的一致通常依赖于量化奖励，这些奖励模糊了哪些目标驱动训练信号。我们引入QA-LIGN，它通过结构化自然语言程序将单一奖励分解为可解释的特定于原则的评估。模型通过起草、评论和修改管道进行学习，其中针对主题的象征性评估为GRPO培训期间的初始和修改响应提供透明的反馈。应用于未经审查的Llama-3.1- 8B-Direct，QA-LIGN将攻击成功率降低高达68.7%，同时保持0.67%的错误拒绝率，实现了帕累托最佳安全帮助性能，并在同等培训的情况下优于DPO和GRPO。这些结果表明，使奖励信号可解释和模块化可以提高对齐有效性，这表明透明度增强了LLM的安全性。



## **35. AIP: Subverting Retrieval-Augmented Generation via Adversarial Instructional Prompt**

AIP：通过对抗性指令提示颠覆检索增强生成 cs.CV

Accepted at EMNLP 2025 Conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15159v1) [paper-pdf](http://arxiv.org/pdf/2509.15159v1)

**Authors**: Saket S. Chaturvedi, Gaurav Bagwe, Lan Zhang, Xiaoyong Yuan

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by retrieving relevant documents from external sources to improve factual accuracy and verifiability. However, this reliance introduces new attack surfaces within the retrieval pipeline, beyond the LLM itself. While prior RAG attacks have exposed such vulnerabilities, they largely rely on manipulating user queries, which is often infeasible in practice due to fixed or protected user inputs. This narrow focus overlooks a more realistic and stealthy vector: instructional prompts, which are widely reused, publicly shared, and rarely audited. Their implicit trust makes them a compelling target for adversaries to manipulate RAG behavior covertly.   We introduce a novel attack for Adversarial Instructional Prompt (AIP) that exploits adversarial instructional prompts to manipulate RAG outputs by subtly altering retrieval behavior. By shifting the attack surface to the instructional prompts, AIP reveals how trusted yet seemingly benign interface components can be weaponized to degrade system integrity. The attack is crafted to achieve three goals: (1) naturalness, to evade user detection; (2) utility, to encourage use of prompts; and (3) robustness, to remain effective across diverse query variations. We propose a diverse query generation strategy that simulates realistic linguistic variation in user queries, enabling the discovery of prompts that generalize across paraphrases and rephrasings. Building on this, a genetic algorithm-based joint optimization is developed to evolve adversarial prompts by balancing attack success, clean-task utility, and stealthiness. Experimental results show that AIP achieves up to 95.23% ASR while preserving benign functionality. These findings uncover a critical and previously overlooked vulnerability in RAG systems, emphasizing the need to reassess the shared instructional prompts.

摘要: 检索增强生成（RAG）通过从外部源检索相关文档来增强大型语言模型（LLM），以提高事实准确性和可验证性。然而，这种依赖在LLM本身之外的检索管道中引入了新的攻击表面。虽然之前的RAG攻击已经暴露了此类漏洞，但它们在很大程度上依赖于操纵用户查询，而由于用户输入固定或受保护，这在实践中通常是不可行的。这种狭隘的焦点忽视了一个更现实、更隐蔽的载体：教学提示，它们被广泛重复使用、公开共享，而且很少审计。他们的隐性信任使他们成为对手秘密操纵RAG行为的引人注目的目标。   我们引入了一种针对对抗性教学提示（AIP）的新型攻击，该攻击利用对抗性教学提示通过微妙地改变检索行为来操纵RAG输出。通过将攻击面转移到指令提示，AIP揭示了如何将可信但看似良性的接口组件武器化以降低系统完整性。该攻击旨在实现三个目标：（1）自然性，以逃避用户检测;（2）实用性，以鼓励使用提示;（3）稳健性，以在不同的查询变体中保持有效。我们提出了一种多样化的查询生成策略，该策略模拟用户查询中现实的语言变化，从而能够发现在重述和改写中进行概括的提示。在此基础上，开发了基于遗传算法的联合优化，通过平衡攻击成功、干净任务效用和隐蔽性来进化对抗提示。实验结果表明，AIP在保持良性功能的同时实现了高达95.23%的ASB。这些发现揭示了RAG系统中一个以前被忽视的关键漏洞，强调需要重新评估共享的教学提示。



## **36. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2409.13174v3) [paper-pdf](http://arxiv.org/pdf/2409.13174v3)

**Authors**: Hao Cheng, Erjia Xiao, Yichi Wang, Chengyuan Yu, Mengshu Sun, Qiang Zhang, Yijie Guo, Kaidi Xu, Jize Zhang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.

摘要: 最近，在多模式大型语言模型（MLLM）进步的推动下，人们提出了视觉语言动作模型（VLAM），以在机器人操纵任务的开放词汇场景中实现更好的性能。由于操纵任务涉及与物理世界的直接互动，因此在执行该任务期间确保稳健性和安全性始终是一个非常关键的问题。本文通过综合当前对MLLM的安全性研究以及物理世界中操纵任务的具体应用场景，对VLAM在潜在物理威胁面前进行了全面评估。具体来说，我们提出了物理脆弱性评估管道（PVEP），它可以整合尽可能多的视觉模式物理威胁，以评估VLAM的物理稳健性。PVEP中的物理威胁具体包括分发外、基于印刷术的视觉提示和对抗性补丁攻击。通过比较VLAM受到攻击前后的性能波动，我们提供了VLAM如何响应不同物理威胁的可概括的\textBF{\textit{Analyses}。



## **37. Sentinel Agents for Secure and Trustworthy Agentic AI in Multi-Agent Systems**

在多代理系统中实现安全且值得信赖的大型人工智能的哨兵代理 cs.AI

25 pages, 12 figures

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14956v1) [paper-pdf](http://arxiv.org/pdf/2509.14956v1)

**Authors**: Diego Gosmar, Deborah A. Dahl

**Abstract**: This paper proposes a novel architectural framework aimed at enhancing security and reliability in multi-agent systems (MAS). A central component of this framework is a network of Sentinel Agents, functioning as a distributed security layer that integrates techniques such as semantic analysis via large language models (LLMs), behavioral analytics, retrieval-augmented verification, and cross-agent anomaly detection. Such agents can potentially oversee inter-agent communications, identify potential threats, enforce privacy and access controls, and maintain comprehensive audit records. Complementary to the idea of Sentinel Agents is the use of a Coordinator Agent. The Coordinator Agent supervises policy implementation, and manages agent participation. In addition, the Coordinator also ingests alerts from Sentinel Agents. Based on these alerts, it can adapt policies, isolate or quarantine misbehaving agents, and contain threats to maintain the integrity of the MAS ecosystem. This dual-layered security approach, combining the continuous monitoring of Sentinel Agents with the governance functions of Coordinator Agents, supports dynamic and adaptive defense mechanisms against a range of threats, including prompt injection, collusive agent behavior, hallucinations generated by LLMs, privacy breaches, and coordinated multi-agent attacks. In addition to the architectural design, we present a simulation study where 162 synthetic attacks of different families (prompt injection, hallucination, and data exfiltration) were injected into a multi-agent conversational environment. The Sentinel Agents successfully detected the attack attempts, confirming the practical feasibility of the proposed monitoring approach. The framework also offers enhanced system observability, supports regulatory compliance, and enables policy evolution over time.

摘要: 本文提出了一种新颖的架构框架，旨在增强多代理系统（MAS）的安全性和可靠性。该框架的核心组件是Sentinel Agents网络，充当分布式安全层，集成了通过大型语言模型（LLM）进行的语义分析、行为分析、检索增强验证和跨代理异常检测等技术。此类代理可以监督代理间的通信、识别潜在威胁、实施隐私和访问控制以及维护全面的审计记录。对哨兵代理想法的补充是协调代理的使用。协调员代理监督政策实施并管理代理参与。此外，协调员还接收来自哨兵特工的警报。基于这些警报，它可以调整政策、隔离或隔离行为不端的代理，并遏制威胁以维护MAS生态系统的完整性。这种双层安全方法将哨兵代理的持续监控与协调代理的治理功能相结合，支持针对一系列威胁的动态和自适应防御机制，包括即时注入、串通代理行为、LLM产生的幻觉、隐私泄露和协调多代理攻击。除了架构设计之外，我们还进行了一项模拟研究，其中将不同家庭的162种合成攻击（即时注射、幻觉和数据泄露）注入到多智能体对话环境中。哨兵特工成功检测到攻击企图，证实了拟议监控方法的实际可行性。该框架还提供增强的系统可观察性、支持监管合规性并使政策能够随着时间的推移而演变。



## **38. Dr. Jekyll and Mr. Hyde: Two Faces of LLMs**

杰基尔博士和海德先生：法学硕士的两面 cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2312.03853v6) [paper-pdf](http://arxiv.org/pdf/2312.03853v6)

**Authors**: Matteo Gioele Collu, Tom Janssen-Groesbeek, Stefanos Koffas, Mauro Conti, Stjepan Picek

**Abstract**: Large Language Models (LLMs) are being integrated into applications such as chatbots or email assistants. To prevent improper responses, safety mechanisms, such as Reinforcement Learning from Human Feedback (RLHF), are implemented in them. In this work, we bypass these safety measures for ChatGPT, Gemini, and Deepseek by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. First, we create elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are provided, making it possible to obtain unauthorized, illegal, or harmful information when querying ChatGPT, Gemini, and Deepseek. We show that these chatbots are vulnerable to this attack by getting dangerous information for 40 out of 40 illicit questions in GPT-4.1-mini, Gemini-1.5-flash, 39 out of 40 in GPT-4o-mini, 38 out of 40 in GPT-3.5-turbo, and 2 out of 2 cases in Gemini-2.5-flash and DeepSeek V3. The attack can be carried out manually or automatically using a support LLM, and has proven effective against models deployed between 2023 and 2025.

摘要: 大型语言模型（LLM）正在集成到聊天机器人或电子邮件助理等应用程序中。为了防止不当响应，其中实施了安全机制，例如来自人类反馈的强化学习（RL HF）。在这项工作中，我们绕过了ChatGPT、Gemini和Deepseek的这些安全措施，让它们模仿具有与诚实助手不一致的性格特征的复杂人物角色。首先，我们创建这些角色的详细传记，然后在与相同聊天机器人的新会话中使用它。然后，我们的对话遵循角色扮演风格，以引发被禁止的回应。使用角色，我们表明提供了禁止的响应，从而在查询ChatGPT、Gemini和Deepseek时可以获得未经授权的、非法的或有害的信息。我们表明，这些聊天机器人容易受到这种攻击，因为它们在GPT-4.1-mini、Gemini-1.5-Flash中获取了40个非法问题的危险信息，GPT-4 o-mini中获取了40个非法问题的危险信息，GPT-3.5-Turbo中获取了40个非法问题的危险信息，而Gemini-2.5-Flash和DeepSeek V3中的2个。该攻击可以使用支持LLM手动或自动执行，并已被证明对2023年至2025年间部署的模型有效。



## **39. MUSE: MCTS-Driven Red Teaming Framework for Enhanced Multi-Turn Dialogue Safety in Large Language Models**

MUSE：MCTS驱动的红色团队框架，用于增强大型语言模型中的多回合对话安全性 cs.CL

EMNLP 2025 main conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14651v1) [paper-pdf](http://arxiv.org/pdf/2509.14651v1)

**Authors**: Siyu Yan, Long Zeng, Xuecheng Wu, Chengcheng Han, Kongcheng Zhang, Chong Peng, Xuezhi Cao, Xunliang Cai, Chenjuan Guo

**Abstract**: As large language models~(LLMs) become widely adopted, ensuring their alignment with human values is crucial to prevent jailbreaks where adversaries manipulate models to produce harmful content. While most defenses target single-turn attacks, real-world usage often involves multi-turn dialogues, exposing models to attacks that exploit conversational context to bypass safety measures. We introduce MUSE, a comprehensive framework tackling multi-turn jailbreaks from both attack and defense angles. For attacks, we propose MUSE-A, a method that uses frame semantics and heuristic tree search to explore diverse semantic trajectories. For defense, we present MUSE-D, a fine-grained safety alignment approach that intervenes early in dialogues to reduce vulnerabilities. Extensive experiments on various models show that MUSE effectively identifies and mitigates multi-turn vulnerabilities. Code is available at \href{https://github.com/yansiyu02/MUSE}{https://github.com/yansiyu02/MUSE}.

摘要: 随着大型语言模型（LLM）的广泛采用，确保它们与人类价值观保持一致对于防止对手操纵模型产生有害内容的越狱至关重要。虽然大多数防御措施都针对单轮攻击，但现实世界的使用通常涉及多轮对话，从而使模型暴露于利用对话上下文绕过安全措施的攻击中。我们引入了MUSE，这是一个从攻击和防御角度解决多回合越狱的综合框架。对于攻击，我们提出了MUE-A，这是一种使用框架语义和启发式树搜索来探索不同的语义轨迹的方法。对于防御，我们提出了MUE-D，这是一种细粒度的安全调整方法，可以在对话中早期干预以减少漏洞。对各种模型的广泛实验表明，MUSE可以有效识别和缓解多回合漏洞。代码可访问\href{https：//github.com/yansiyu02/MUSE}{https：//github.com/yansiyu02/MUSE}。



## **40. Enterprise AI Must Enforce Participant-Aware Access Control**

企业人工智能必须强制执行用户感知访问控制 cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14608v1) [paper-pdf](http://arxiv.org/pdf/2509.14608v1)

**Authors**: Shashank Shreedhar Bhatt, Tanmay Rajore, Khushboo Aggarwal, Ganesh Ananthanarayanan, Ranveer Chandra, Nishanth Chandran, Suyash Choudhury, Divya Gupta, Emre Kiciman, Sumit Kumar Pandey, Srinath Setty, Rahul Sharma, Teijia Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in enterprise settings where they interact with multiple users and are trained or fine-tuned on sensitive internal data. While fine-tuning enhances performance by internalizing domain knowledge, it also introduces a critical security risk: leakage of confidential training data to unauthorized users. These risks are exacerbated when LLMs are combined with Retrieval-Augmented Generation (RAG) pipelines that dynamically fetch contextual documents at inference time.   We demonstrate data exfiltration attacks on AI assistants where adversaries can exploit current fine-tuning and RAG architectures to leak sensitive information by leveraging the lack of access control enforcement. We show that existing defenses, including prompt sanitization, output filtering, system isolation, and training-level privacy mechanisms, are fundamentally probabilistic and fail to offer robust protection against such attacks.   We take the position that only a deterministic and rigorous enforcement of fine-grained access control during both fine-tuning and RAG-based inference can reliably prevent the leakage of sensitive data to unauthorized recipients.   We introduce a framework centered on the principle that any content used in training, retrieval, or generation by an LLM is explicitly authorized for \emph{all users involved in the interaction}. Our approach offers a simple yet powerful paradigm shift for building secure multi-user LLM systems that are grounded in classical access control but adapted to the unique challenges of modern AI workflows. Our solution has been deployed in Microsoft Copilot Tuning, a product offering that enables organizations to fine-tune models using their own enterprise-specific data.

摘要: 大型语言模型（LLM）越来越多地部署在企业环境中，它们与多个用户交互，并根据敏感的内部数据接受培训或微调。虽然微调通过内化领域知识来提高性能，但它也会带来严重的安全风险：机密培训数据泄露给未经授权的用户。当LLM与在推理时动态获取上下文文档的检索增强生成（RAG）管道相结合时，这些风险就会加剧。   我们展示了对人工智能助手的数据泄露攻击，其中对手可以利用当前的微调和RAG架构，通过利用访问控制强制执行的缺乏来泄露敏感信息。我们表明，现有的防御措施，包括即时清理、输出过滤、系统隔离和训练级隐私机制，从根本上来说是概率性的，无法提供针对此类攻击的强有力保护。   我们的立场是，只有在微调和基于RAG的推理期间确定性且严格地执行细粒度的访问控制，才能可靠地防止敏感数据泄露给未经授权的接收者。   我们引入了一个框架，其核心原则是LLM在训练、检索或生成中使用的任何内容都被明确授权给\{参与交互的所有用户}。我们的方法为构建安全的多用户LLM系统提供了简单而强大的范式转变，该系统基于经典的访问控制，但适应现代人工智能工作流程的独特挑战。我们的解决方案已部署在Microsoft Copilot Tuning中，这是一种产品，使组织能够使用自己的企业特定数据微调模型。



## **41. Reconstruction of Differentially Private Text Sanitization via Large Language Models**

通过大语言模型重建差异私人文本清理 cs.CR

RAID-2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2410.12443v3) [paper-pdf](http://arxiv.org/pdf/2410.12443v3)

**Authors**: Shuchao Pang, Zhigang Lu, Haichen Wang, Peng Fu, Yongbin Zhou, Minhui Xue

**Abstract**: Differential privacy (DP) is the de facto privacy standard against privacy leakage attacks, including many recently discovered ones against large language models (LLMs). However, we discovered that LLMs could reconstruct the altered/removed privacy from given DP-sanitized prompts. We propose two attacks (black-box and white-box) based on the accessibility to LLMs and show that LLMs could connect the pair of DP-sanitized text and the corresponding private training data of LLMs by giving sample text pairs as instructions (in the black-box attacks) or fine-tuning data (in the white-box attacks). To illustrate our findings, we conduct comprehensive experiments on modern LLMs (e.g., LLaMA-2, LLaMA-3, ChatGPT-3.5, ChatGPT-4, ChatGPT-4o, Claude-3, Claude-3.5, OPT, GPT-Neo, GPT-J, Gemma-2, and Pythia) using commonly used datasets (such as WikiMIA, Pile-CC, and Pile-Wiki) against both word-level and sentence-level DP. The experimental results show promising recovery rates, e.g., the black-box attacks against the word-level DP over WikiMIA dataset gave 72.18% on LLaMA-2 (70B), 82.39% on LLaMA-3 (70B), 75.35% on Gemma-2, 91.2% on ChatGPT-4o, and 94.01% on Claude-3.5 (Sonnet). More urgently, this study indicates that these well-known LLMs have emerged as a new security risk for existing DP text sanitization approaches in the current environment.

摘要: 差异隐私（DP）是针对隐私泄露攻击的事实上的隐私标准，包括最近发现的许多针对大型语言模型（LLM）的攻击。然而，我们发现LLM可以从给定的DP消毒提示中重建更改/删除的隐私。我们基于LLM的可访问性提出了两种攻击（黑匣子和白盒），并表明LLM可以通过提供样本文本对作为指令（在黑匣子攻击中）或微调数据（在白盒攻击中）来连接DP清理文本对和LLM的相应私人训练数据。为了说明我们的发现，我们对现代LLM进行了全面的实验（例如，LLaMA-2、LLaMA-3、ChatGPT-3.5、ChatGPT-4、ChatGPT-4 o、Claude-3、Claude-3.5、OPT、GPT-Neo、GPT-J、Gemma-2和Pythia）针对单词级和业务级DP使用常用数据集（例如WikiMIA、Pile-CC和Pile-iki）。实验结果显示出有希望的回收率，例如，针对WikiMIA数据集的词级DP的黑匣子攻击在LLaMA-2（70 B）上为72.18%，在LLaMA-3（70 B）上为82.39%，在Gemma-2上为75.35%，在ChatGPT-4 o上为91.2%，在Claude-3.5（十四行诗）上为94.01%。更紧迫的是，这项研究表明，这些著名的LLM已成为当前环境下现有DP文本清理方法的新安全风险。



## **42. SynBench: A Benchmark for Differentially Private Text Generation**

SynBench：差异私密文本生成的基准 cs.AI

15 pages

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14594v1) [paper-pdf](http://arxiv.org/pdf/2509.14594v1)

**Authors**: Yidan Sun, Viktor Schlegel, Srinivasan Nandakumar, Iqra Zahid, Yuping Wu, Yulong Wu, Hao Li, Jie Zhang, Warren Del-Pinto, Goran Nenadic, Siew Kei Lam, Anil Anthony Bharath

**Abstract**: Data-driven decision support in high-stakes domains like healthcare and finance faces significant barriers to data sharing due to regulatory, institutional, and privacy concerns. While recent generative AI models, such as large language models, have shown impressive performance in open-domain tasks, their adoption in sensitive environments remains limited by unpredictable behaviors and insufficient privacy-preserving datasets for benchmarking. Existing anonymization methods are often inadequate, especially for unstructured text, as redaction and masking can still allow re-identification. Differential Privacy (DP) offers a principled alternative, enabling the generation of synthetic data with formal privacy assurances. In this work, we address these challenges through three key contributions. First, we introduce a comprehensive evaluation framework with standardized utility and fidelity metrics, encompassing nine curated datasets that capture domain-specific complexities such as technical jargon, long-context dependencies, and specialized document structures. Second, we conduct a large-scale empirical study benchmarking state-of-the-art DP text generation methods and LLMs of varying sizes and different fine-tuning strategies, revealing that high-quality domain-specific synthetic data generation under DP constraints remains an unsolved challenge, with performance degrading as domain complexity increases. Third, we develop a membership inference attack (MIA) methodology tailored for synthetic text, providing first empirical evidence that the use of public datasets - potentially present in pre-training corpora - can invalidate claimed privacy guarantees. Our findings underscore the urgent need for rigorous privacy auditing and highlight persistent gaps between open-domain and specialist evaluations, informing responsible deployment of generative AI in privacy-sensitive, high-stakes settings.

摘要: 由于监管、机构和隐私问题，医疗保健和金融等高风险领域的数据驱动决策支持在数据共享方面面临巨大障碍。虽然最近的生成性人工智能模型（例如大型语言模型）在开放领域任务中表现出了令人印象深刻的性能，但它们在敏感环境中的采用仍然受到不可预测的行为和用于基准测试的隐私保护数据集不足的限制。现有的匿名化方法通常不充分，尤其是对于非结构化文本，因为编辑和掩蔽仍然可以允许重新识别。差异隐私（DP）提供了一种有原则的替代方案，可以生成具有正式隐私保证的合成数据。在这项工作中，我们通过三项关键贡献来应对这些挑战。首先，我们引入了一个具有标准化效用和保真度指标的全面评估框架，其中包含九个精心策划的数据集，这些数据集捕捉特定领域的复杂性，例如技术行话、长上下文依赖性和专业文档结构。其次，我们进行了一项大规模的实证研究，对最先进的DP文本生成方法和不同规模和不同微调策略的LLM进行了基准测试，揭示了DP约束下的高质量特定领域的合成数据生成仍然是一个尚未解决的挑战，随着领域复杂性的增加，性能会下降。第三，我们开发了一种专为合成文本量身定制的成员资格推理攻击（MIA）方法，提供了第一个经验证据，证明使用公共数据集（可能存在于预训练库中）可以使声称的隐私保证无效。我们的研究结果强调了严格的隐私审计的迫切需要，并强调了开放领域和专业评估之间持续存在的差距，为在隐私敏感、高风险的环境中负责任地部署生成性人工智能提供信息。



## **43. Unique Security and Privacy Threats of Large Language Models: A Comprehensive Survey**

大型语言模型的独特安全和隐私威胁：全面调查 cs.CR

35 pages, 9 tables, 12 figures. To appear in ACM Computing Surveys  (CSUR), 2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2406.07973v3) [paper-pdf](http://arxiv.org/pdf/2406.07973v3)

**Authors**: Shang Wang, Tianqing Zhu, Bo Liu, Ming Ding, Dayong Ye, Wanlei Zhou, Philip S. Yu

**Abstract**: With the rapid development of artificial intelligence, large language models (LLMs) have made remarkable advancements in natural language processing. These models are trained on vast datasets to exhibit powerful language understanding and generation capabilities across various applications, including chatbots, and agents. However, LLMs have revealed a variety of privacy and security issues throughout their life cycle, drawing significant academic and industrial attention. Moreover, the risks faced by LLMs differ significantly from those encountered by traditional language models. Given that current surveys lack a clear taxonomy of unique threat models across diverse scenarios, we emphasize the unique privacy and security threats associated with four specific scenarios: pre-training, fine-tuning, deployment, and LLM-based agents. Addressing the characteristics of each risk, this survey outlines and analyzes potential countermeasures. Research on attack and defense situations can offer feasible research directions, enabling more areas to benefit from LLMs.

摘要: 随着人工智能的快速发展，大型语言模型（LLM）在自然语言处理方面取得了显着进步。这些模型在大量数据集上训练，以在包括聊天机器人和代理在内的各种应用程序中展现强大的语言理解和生成能力。然而，LLM在其整个生命周期中揭示了各种隐私和安全问题，引起了学术界和行业的高度关注。此外，LLM面临的风险与传统语言模型遇到的风险显着不同。鉴于当前的调查缺乏对不同场景中的独特威胁模型的明确分类，我们强调与四种特定场景相关的独特隐私和安全威胁：预训练、微调、部署和基于LLM的代理。本调查针对每种风险的特征，概述并分析了潜在的应对措施。对攻击和防御情况的研究可以提供可行的研究方向，使更多领域能够从LLM中受益。



## **44. LLM Jailbreak Detection for (Almost) Free!**

LLM越狱检测（几乎）免费！ cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14558v1) [paper-pdf](http://arxiv.org/pdf/2509.14558v1)

**Authors**: Guorui Chen, Yifan Xia, Xiaojun Jia, Zhijiang Li, Philip Torr, Jindong Gu

**Abstract**: Large language models (LLMs) enhance security through alignment when widely used, but remain susceptible to jailbreak attacks capable of producing inappropriate content. Jailbreak detection methods show promise in mitigating jailbreak attacks through the assistance of other models or multiple model inferences. However, existing methods entail significant computational costs. In this paper, we first present a finding that the difference in output distributions between jailbreak and benign prompts can be employed for detecting jailbreak prompts. Based on this finding, we propose a Free Jailbreak Detection (FJD) which prepends an affirmative instruction to the input and scales the logits by temperature to further distinguish between jailbreak and benign prompts through the confidence of the first token. Furthermore, we enhance the detection performance of FJD through the integration of virtual instruction learning. Extensive experiments on aligned LLMs show that our FJD can effectively detect jailbreak prompts with almost no additional computational costs during LLM inference.

摘要: 大型语言模型（LLM）在广泛使用时通过对齐来增强安全性，但仍然容易受到能够产生不当内容的越狱攻击。越狱检测方法在通过其他模型或多个模型推断的帮助减轻越狱攻击方面表现出了希望。然而，现有方法需要大量的计算成本。在本文中，我们首先提出了一个发现，即越狱和良性提示之间的输出分布差异可以用于检测越狱提示。基于这一发现，我们提出了一种免费越狱检测（FJD），它在输入中预先添加肯定指令，并通过温度缩放逻辑比特，以通过第一个令牌的置信度进一步区分越狱和良性提示。此外，我们还通过集成虚拟教学学习来提高FJD的检测性能。对对齐LLM的大量实验表明，我们的FJD可以有效地检测越狱提示，而在LLM推理期间几乎没有额外的计算成本。



## **45. GRADA: Graph-based Reranking against Adversarial Documents Attack**

GRADA：基于图的重新排名对抗文档攻击 cs.IR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2505.07546v3) [paper-pdf](http://arxiv.org/pdf/2505.07546v3)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.

摘要: 检索增强生成（RAG）框架通过集成来自检索文档的外部知识来提高大型语言模型（LLM）的准确性，从而克服模型静态内在知识的局限性。然而，这些系统很容易受到对抗性攻击，这些攻击通过引入对抗性但在语义上与查询相似的文档来操纵检索过程。值得注意的是，虽然这些对抗性文档类似于查询，但它们与检索集中的良性文档表现出弱的相似性。因此，我们提出了一个简单而有效的基于图形的对抗性文档攻击重新排名（GRADA）框架，旨在保留检索质量，同时显着降低对手的成功。我们的研究通过在五个LLM上进行的实验来评估我们的方法的有效性：GPT-3.5-Turbo，GPT-4 o，Llama3.1-8b，Llama3.1- 70 b和Qwen2.5- 7 b。我们使用三个数据集来评估性能，来自Natural Questions数据集的结果表明攻击成功率降低了80%，同时保持了最小的准确性损失。



## **46. Benchmarking Large Language Models for Cryptanalysis and Side-Channel Vulnerabilities**

针对加密分析和侧通道漏洞对大型语言模型进行基准测试 cs.CL

EMNLP'25 Findings

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2505.24621v2) [paper-pdf](http://arxiv.org/pdf/2505.24621v2)

**Authors**: Utsav Maskey, Chencheng Zhu, Usman Naseem

**Abstract**: Recent advancements in large language models (LLMs) have transformed natural language understanding and generation, leading to extensive benchmarking across diverse tasks. However, cryptanalysis - a critical area for data security and its connection to LLMs' generalization abilities - remains underexplored in LLM evaluations. To address this gap, we evaluate the cryptanalytic potential of state-of-the-art LLMs on ciphertexts produced by a range of cryptographic algorithms. We introduce a benchmark dataset of diverse plaintexts, spanning multiple domains, lengths, writing styles, and topics, paired with their encrypted versions. Using zero-shot and few-shot settings along with chain-of-thought prompting, we assess LLMs' decryption success rate and discuss their comprehension abilities. Our findings reveal key insights into LLMs' strengths and limitations in side-channel scenarios and raise concerns about their susceptibility to under-generalization-related attacks. This research highlights the dual-use nature of LLMs in security contexts and contributes to the ongoing discussion on AI safety and security.

摘要: 大型语言模型（LLM）的最新进展已经改变了自然语言的理解和生成，导致了跨各种任务的广泛基准测试。然而，密码分析-数据安全的一个关键领域及其与LLM泛化能力的联系-在LLM评估中仍然没有得到充分的探索。为了解决这一差距，我们评估的密码分析潜力的国家的最先进的LLM的密文产生的一系列密码算法。我们介绍了一个基准数据集的不同明文，跨越多个领域，长度，写作风格和主题，与他们的加密版本配对。使用零镜头和少镜头设置以及思想链提示，我们评估LLM的解密成功率并讨论他们的理解能力。我们的研究结果揭示了对LLM在侧通道场景中的优势和局限性的关键见解，并引发了人们对它们容易受到归因不足相关攻击的担忧。这项研究强调了LLM在安全环境中的双重用途性质，并有助于正在进行的关于人工智能安全性的讨论。



## **47. Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs**

评估和改进LLM生成的安全攻击检测器的鲁棒性 cs.SE

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2411.18216v2) [paper-pdf](http://arxiv.org/pdf/2411.18216v2)

**Authors**: Samuele Pasini, Jinhan Kim, Tommaso Aiello, Rocio Cabrera Lozoya, Antonino Sabetta, Paolo Tonella

**Abstract**: Large Language Models (LLMs) are increasingly used in software development to generate functions, such as attack detectors, that implement security requirements. A key challenge is ensuring the LLMs have enough knowledge to address specific security requirements, such as information about existing attacks. For this, we propose an approach integrating Retrieval Augmented Generation (RAG) and Self-Ranking into the LLM pipeline. RAG enhances the robustness of the output by incorporating external knowledge sources, while the Self-Ranking technique, inspired by the concept of Self-Consistency, generates multiple reasoning paths and creates ranks to select the most robust detector. Our extensive empirical study targets code generated by LLMs to detect two prevalent injection attacks in web security: Cross-Site Scripting (XSS) and SQL injection (SQLi). Results show a significant improvement in detection performance while employing RAG and Self-Ranking, with an increase of up to 71%pt (on average 37%pt) and up to 43%pt (on average 6%pt) in the F2-Score for XSS and SQLi detection, respectively.

摘要: 大型语言模型（LLM）越来越多地用于软件开发来生成实现安全要求的函数，例如攻击检测器。一个关键挑战是确保LLM拥有足够的知识来满足特定的安全要求，例如有关现有攻击的信息。为此，我们提出了一种将检索增强生成（RAG）和自我排名集成到LLM管道中的方法。RAG通过整合外部知识源来增强输出的稳健性，而自排名技术则受到自一致性概念的启发，生成多个推理路径并创建排名来选择最稳健的检测器。我们广泛的实证研究针对LLM生成的代码，以检测网络安全中两种普遍的注入攻击：跨站点脚本（XSS）和SQL注入（SQLi）。结果显示，采用RAG和Self-Ranking时检测性能显着提高，XSS和SQLi检测的F2评分分别增加了71%pt（平均37%pt）和43%pt（平均6%pt）。



## **48. CyberLLMInstruct: A Pseudo-malicious Dataset Revealing Safety-performance Trade-offs in Cyber Security LLM Fine-tuning**

CyberLLMDirecct：揭示网络安全中安全性能权衡的伪恶意数据集LLM微调 cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2503.09334v3) [paper-pdf](http://arxiv.org/pdf/2503.09334v3)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents both opportunities and critical safety risks. We introduce CyberLLMInstruct, a dataset of 54,928 pseudo-malicious instruction-response pairs spanning cyber security tasks including malware analysis, phishing simulations, and zero-day vulnerabilities. Our comprehensive evaluation using seven open-source LLMs reveals a critical trade-off: while fine-tuning improves cyber security task performance (achieving up to 92.50% accuracy on CyberMetric), it severely compromises safety resilience across all tested models and attack vectors (e.g., Llama 3.1 8B's security score against prompt injection drops from 0.95 to 0.15). The dataset incorporates diverse sources including CTF challenges, academic papers, industry reports, and CVE databases to ensure comprehensive coverage of cyber security domains. Our findings highlight the unique challenges of securing LLMs in adversarial domains and establish the critical need for developing fine-tuning methodologies that balance performance gains with safety preservation in security-sensitive domains.

摘要: 将大型语言模型（LLM）集成到网络安全应用程序中既带来了机遇，也带来了严重的安全风险。我们引入CyberLLMCinsert，这是一个由54，928个伪恶意描述-响应对组成的数据集，涵盖网络安全任务，包括恶意软件分析、网络钓鱼模拟和零日漏洞。我们使用七个开源LLM进行的全面评估揭示了一个关键的权衡：虽然微调可以提高网络安全任务性能（在CyberMetric上实现高达92.50%的准确率），但它严重损害了所有测试模型和攻击载体的安全弹性（例如，Lama 3.1 8B对立即注射的安全评分从0.95下降到0.15）。该数据集融合了多种来源，包括CTF挑战、学术论文、行业报告和UTE数据库，以确保网络安全领域的全面覆盖。我们的研究结果强调了在对抗性领域中保护LLM的独特挑战，并确定了开发微调方法的迫切需求，该方法在安全敏感领域中平衡性能收益与安全保护。



## **49. Do LLMs Align Human Values Regarding Social Biases? Judging and Explaining Social Biases with LLMs**

法学硕士是否在社会偏见方面与人类价值观保持一致？利用法学硕士判断和解释社会偏见 cs.CL

38 pages, 31 figures

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.13869v1) [paper-pdf](http://arxiv.org/pdf/2509.13869v1)

**Authors**: Yang Liu, Chenhui Chu

**Abstract**: Large language models (LLMs) can lead to undesired consequences when misaligned with human values, especially in scenarios involving complex and sensitive social biases. Previous studies have revealed the misalignment of LLMs with human values using expert-designed or agent-based emulated bias scenarios. However, it remains unclear whether the alignment of LLMs with human values differs across different types of scenarios (e.g., scenarios containing negative vs. non-negative questions). In this study, we investigate the alignment of LLMs with human values regarding social biases (HVSB) in different types of bias scenarios. Through extensive analysis of 12 LLMs from four model families and four datasets, we demonstrate that LLMs with large model parameter scales do not necessarily have lower misalignment rate and attack success rate. Moreover, LLMs show a certain degree of alignment preference for specific types of scenarios and the LLMs from the same model family tend to have higher judgment consistency. In addition, we study the understanding capacity of LLMs with their explanations of HVSB. We find no significant differences in the understanding of HVSB across LLMs. We also find LLMs prefer their own generated explanations. Additionally, we endow smaller language models (LMs) with the ability to explain HVSB. The generation results show that the explanations generated by the fine-tuned smaller LMs are more readable, but have a relatively lower model agreeability.

摘要: 大型语言模型（LLM）与人类价值观不一致时可能会导致不良后果，尤其是在涉及复杂和敏感的社会偏见的场景中。之前的研究使用专家设计或基于代理的模拟偏见场景揭示了LLM与人类价值观的不一致。然而，目前尚不清楚LLM与人类价值观的一致是否在不同类型的场景中有所不同（例如，包含负面问题与非负面问题的场景）。在这项研究中，我们调查了在不同类型的偏见场景中，LLM与人类社会偏见（HCSB）价值观的一致性。通过对来自四个模型系列和四个数据集的12个LLM的广泛分析，我们证明具有大模型参数规模的LLM不一定具有较低的失准率和攻击成功率。此外，LLM对特定类型的场景表现出一定程度的一致偏好，并且来自同一模型家族的LLM往往具有更高的判断一致性。此外，我们还研究了LLM的理解能力及其对HCSB的解释。我们发现各LLM对HCSB的理解没有显着差异。我们还发现LLM更喜欢他们自己生成的解释。此外，我们赋予较小的语言模型（LM）解释HDSB的能力。生成结果表明，微调后的较小LM生成的解释更具可读性，但具有相对较低的模型。



## **50. Defending against Indirect Prompt Injection by Instruction Detection**

利用指令检测防御间接提示注入 cs.CR

16 pages, 4 figures

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2505.06311v2) [paper-pdf](http://arxiv.org/pdf/2505.06311v2)

**Authors**: Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, Fangzhao Wu

**Abstract**: The integration of Large Language Models (LLMs) with external sources is becoming increasingly common, with Retrieval-Augmented Generation (RAG) being a prominent example. However, this integration introduces vulnerabilities of Indirect Prompt Injection (IPI) attacks, where hidden instructions embedded in external data can manipulate LLMs into executing unintended or harmful actions. We recognize that IPI attacks fundamentally rely on the presence of instructions embedded within external content, which can alter the behavioral states of LLMs. Can the effective detection of such state changes help us defend against IPI attacks? In this paper, we propose InstructDetector, a novel detection-based approach that leverages the behavioral states of LLMs to identify potential IPI attacks. Specifically, we demonstrate the hidden states and gradients from intermediate layers provide highly discriminative features for instruction detection. By effectively combining these features, InstructDetector achieves a detection accuracy of 99.60% in the in-domain setting and 96.90% in the out-of-domain setting, and reduces the attack success rate to just 0.03% on the BIPIA benchmark. The code is publicly available at https://github.com/MYVAE/Instruction-detection.

摘要: 大型语言模型（LLM）与外部源的集成变得越来越常见，检索增强生成（RAG）就是一个突出的例子。然而，此集成引入了间接提示注入（IPI）攻击的漏洞，其中嵌入外部数据中的隐藏指令可以操纵LLM执行无意或有害的操作。我们认识到，IPI攻击从根本上依赖于外部内容中嵌入的指令的存在，这些指令可以改变LLM的行为状态。有效检测此类状态变化能否帮助我们抵御IPI攻击？在本文中，我们提出了DirectDetector，这是一种新型的基于检测的方法，利用LLM的行为状态来识别潜在的IPI攻击。具体来说，我们证明了来自中间层的隐藏状态和梯度为指令检测提供了高度区分性的特征。通过有效结合这些功能，DirectDetector在域内设置中的检测准确率为99.60%，在域外设置中的检测准确率为96.90%，并将BIPIA基准测试中的攻击成功率降低至仅0.03%。该代码可在https://github.com/MYVAE/Instruction-detection上公开获取。



