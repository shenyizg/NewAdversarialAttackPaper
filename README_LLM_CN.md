# Latest Large Language Model Attack Papers
**update at 2025-08-24 09:59:17**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. SDGO: Self-Discrimination-Guided Optimization for Consistent Safety in Large Language Models**

SDGO：自我辨别引导的大型语言模型中一致安全性优化 cs.CL

Accepted by EMNLP 2025, 15 pages, 4 figures, 6 tables

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15648v1) [paper-pdf](http://arxiv.org/pdf/2508.15648v1)

**Authors**: Peng Ding, Wen Sun, Dailin Li, Wei Zou, Jiaming Wang, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs) excel at various natural language processing tasks but remain vulnerable to jailbreaking attacks that induce harmful content generation. In this paper, we reveal a critical safety inconsistency: LLMs can more effectively identify harmful requests as discriminators than defend against them as generators. This insight inspires us to explore aligning the model's inherent discrimination and generation capabilities. To this end, we propose SDGO (Self-Discrimination-Guided Optimization), a reinforcement learning framework that leverages the model's own discrimination capabilities as a reward signal to enhance generation safety through iterative self-improvement. Our method does not require any additional annotated data or external models during the training phase. Extensive experiments demonstrate that SDGO significantly improves model safety compared to both prompt-based and training-based baselines while maintaining helpfulness on general benchmarks. By aligning LLMs' discrimination and generation capabilities, SDGO brings robust performance against out-of-distribution (OOD) jailbreaking attacks. This alignment achieves tighter coupling between these two capabilities, enabling the model's generation capability to be further enhanced with only a small amount of discriminative samples. Our code and datasets are available at https://github.com/NJUNLP/SDGO.

摘要: 大型语言模型（LLM）擅长各种自然语言处理任务，但仍然容易受到导致有害内容生成的越狱攻击。在本文中，我们揭示了一个关键的安全不一致性：LLM可以更有效地识别有害请求作为识别器，而不是作为生成器来防御有害请求。这一见解激励我们探索如何调整模型的固有歧视和生成能力。为此，我们提出了SDGO（自我歧视引导优化），这是一种强化学习框架，它利用模型自身的歧视能力作为奖励信号，通过迭代自我改进来增强发电安全性。我们的方法在训练阶段不需要任何额外的注释数据或外部模型。大量实验表明，与基于预算和基于培训的基线相比，SDGO显着提高了模型安全性，同时保持了一般基准的帮助性。通过协调LLM的区分和生成能力，SDGO为抵御分发外（OOD）越狱攻击带来了强劲的性能。这种对齐实现了这两种功能之间的更紧密耦合，使模型的生成能力能够进一步增强，只需少量的判别样本。我们的代码和数据集可以在https://github.com/NJUNLP/SDGO上找到。



## **2. The Enemy from Within: A Study of Political Delegitimization Discourse in Israeli Political Speech**

来自内部的敌人：以色列政治演讲中政治去合法化话语的研究 cs.CL

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15524v1) [paper-pdf](http://arxiv.org/pdf/2508.15524v1)

**Authors**: Naama Rivlin-Angert, Guy Mor-Lan

**Abstract**: We present the first large-scale computational study of political delegitimization discourse (PDD), defined as symbolic attacks on the normative validity of political entities. We curate and manually annotate a novel Hebrew-language corpus of 10,410 sentences drawn from Knesset speeches (1993-2023), Facebook posts (2018-2021), and leading news outlets, of which 1,812 instances (17.4\%) exhibit PDD and 642 carry additional annotations for intensity, incivility, target type, and affective framing. We introduce a two-stage classification pipeline combining finetuned encoder models and decoder LLMs. Our best model (DictaLM 2.0) attains an F$_1$ of 0.74 for binary PDD detection and a macro-F$_1$ of 0.67 for classification of delegitimization characteristics. Applying this classifier to longitudinal and cross-platform data, we see a marked rise in PDD over three decades, higher prevalence on social media versus parliamentary debate, greater use by male than female politicians, and stronger tendencies among right-leaning actors - with pronounced spikes during election campaigns and major political events. Our findings demonstrate the feasibility and value of automated PDD analysis for understanding democratic discourse.

摘要: 我们对政治去合法性话语（PDD）进行了首次大规模计算研究，PDD被定义为对政治实体规范有效性的象征性攻击。我们策划并手动注释了一个新颖的希伯来语数据库，其中包含10，410个句子，取自以色列议会演讲（1993-2023年）、Facebook帖子（2018-2021年）和领先新闻媒体，其中1，812个实例（17.4%）表现出PDD，642个实例带有强度、礼貌、目标类型和情感框架的额外注释。我们引入了一个两阶段分类流水线，结合了微调编码器模型和解码器LLM。我们的最佳模型（DictaLM 2.0）对于二进制PDD检测，F$_1$为0.74，对于去合法化特征分类，宏F$_1 $为0.67。将这种分类器应用于纵向和跨平台数据，我们看到三十年来PDD显着上升，社交媒体上的流行率高于议会辩论，男性政客的使用率高于女性政客，右倾行为者的倾向更强--在竞选和重大政治活动期间出现明显峰值。我们的研究结果证明了自动PDD分析对于理解民主话语的可行性和价值。



## **3. Reliable Unlearning Harmful Information in LLMs with Metamorphosis Representation Projection**

通过变形表示投影可靠地消除LLM中的有害信息 cs.LG

10 pages, 9 figures, Under review as a full paper at AAAI 2026. A  preliminary version is under review at the NeurIPS 2025 Workshop on Reliable  ML from Unreliable Data

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15449v1) [paper-pdf](http://arxiv.org/pdf/2508.15449v1)

**Authors**: Chengcan Wu, Zeming Wei, Huanran Chen, Yinpeng Dong, Meng Sun

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive performance in various domains and tasks, concerns about their safety are becoming increasingly severe. In particular, since models may store unsafe knowledge internally, machine unlearning has emerged as a representative paradigm to ensure model safety. Existing approaches employ various training techniques, such as gradient ascent and negative preference optimization, in attempts to eliminate the influence of undesired data on target models. However, these methods merely suppress the activation of undesired data through parametric training without completely eradicating its informational traces within the model. This fundamental limitation makes it difficult to achieve effective continuous unlearning, rendering these methods vulnerable to relearning attacks. To overcome these challenges, we propose a Metamorphosis Representation Projection (MRP) approach that pioneers the application of irreversible projection properties to machine unlearning. By implementing projective transformations in the hidden state space of specific network layers, our method effectively eliminates harmful information while preserving useful knowledge. Experimental results demonstrate that our approach enables effective continuous unlearning and successfully defends against relearning attacks, achieving state-of-the-art performance in unlearning effectiveness while preserving natural performance. Our code is available in https://github.com/ChengcanWu/MRP.

摘要: 虽然大型语言模型（LLM）在各个领域和任务中表现出了令人印象深刻的性能，但对其安全性的担忧却变得越来越严重。特别是，由于模型可能会在内部存储不安全的知识，因此机器去学习已成为确保模型安全性的代表性范式。现有的方法采用各种训练技术，例如梯度上升和负偏好优化，试图消除不期望数据对目标模型的影响。然而，这些方法只是通过参数训练抑制不需要数据的激活，而没有完全消除模型中的信息痕迹。这一基本限制使得很难实现有效的连续取消学习，从而使这些方法容易受到重新学习攻击。为了克服这些挑战，我们提出了一种变形表示投影（MRP）方法，该方法开创了将不可逆投影特性应用于机器去学习的先河。通过在特定网络层的隐藏状态空间中实现投影变换，我们的方法有效地消除了有害信息，同时保留了有用的知识。实验结果表明，我们的方法能够实现有效的连续去学习，并成功防御重新学习攻击，在去学习有效性方面实现了最先进的性能，同时保持了自然性能。我们的代码可在https://github.com/ChengcanWu/MRP上找到。



## **4. IPIGuard: A Novel Tool Dependency Graph-Based Defense Against Indirect Prompt Injection in LLM Agents**

IPIGuard：一种新型工具依赖于图形的防御，针对LLM代理中间接即时注入 cs.CR

EMNLP 2025

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15310v1) [paper-pdf](http://arxiv.org/pdf/2508.15310v1)

**Authors**: Hengyu An, Jinghuai Zhang, Tianyu Du, Chunyi Zhou, Qingming Li, Tao Lin, Shouling Ji

**Abstract**: Large language model (LLM) agents are widely deployed in real-world applications, where they leverage tools to retrieve and manipulate external data for complex tasks. However, when interacting with untrusted data sources (e.g., fetching information from public websites), tool responses may contain injected instructions that covertly influence agent behaviors and lead to malicious outcomes, a threat referred to as Indirect Prompt Injection (IPI). Existing defenses typically rely on advanced prompting strategies or auxiliary detection models. While these methods have demonstrated some effectiveness, they fundamentally rely on assumptions about the model's inherent security, which lacks structural constraints on agent behaviors. As a result, agents still retain unrestricted access to tool invocations, leaving them vulnerable to stronger attack vectors that can bypass the security guardrails of the model. To prevent malicious tool invocations at the source, we propose a novel defensive task execution paradigm, called IPIGuard, which models the agents' task execution process as a traversal over a planned Tool Dependency Graph (TDG). By explicitly decoupling action planning from interaction with external data, IPIGuard significantly reduces unintended tool invocations triggered by injected instructions, thereby enhancing robustness against IPI attacks. Experiments on the AgentDojo benchmark show that IPIGuard achieves a superior balance between effectiveness and robustness, paving the way for the development of safer agentic systems in dynamic environments.

摘要: 大型语言模型（LLM）代理广泛部署在现实世界的应用程序中，它们利用工具来检索和操纵复杂任务的外部数据。然而，当与不受信任的数据源（例如，从公共网站获取信息），工具响应可能包含注入的指令，这些指令秘密影响代理行为并导致恶意结果，这种威胁称为间接提示注入（IPI）。现有的防御通常依赖于高级提示策略或辅助检测模型。虽然这些方法已经证明了一定的有效性，但它们从根本上依赖于对模型固有安全性的假设，而该假设缺乏对代理行为的结构性约束。因此，代理仍然保留对工具调用的不受限制的访问权限，这使得它们容易受到更强的攻击载体的攻击，而这些攻击载体可以绕过模型的安全护栏。为了从源头防止恶意工具调用，我们提出了一种新型的防御性任务执行范式，称为IPIGuard，它将代理的任务执行过程建模为对计划的工具依赖图（TDG）的穿越。通过显式地将行动规划与与外部数据的交互脱钩，IPIGuard显着减少了由注入指令触发的意外工具调用，从而增强了针对IPI攻击的鲁棒性。AgentDojo基准测试的实验表明，IPIGuard在有效性和稳健性之间实现了卓越的平衡，为在动态环境中开发更安全的代理系统铺平了道路。



## **5. Adversarial Attacks against Neural Ranking Models via In-Context Learning**

通过上下文学习对神经排名模型的对抗攻击 cs.IR

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15283v1) [paper-pdf](http://arxiv.org/pdf/2508.15283v1)

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke

**Abstract**: While neural ranking models (NRMs) have shown high effectiveness, they remain susceptible to adversarial manipulation. In this work, we introduce Few-Shot Adversarial Prompting (FSAP), a novel black-box attack framework that leverages the in-context learning capabilities of Large Language Models (LLMs) to generate high-ranking adversarial documents. Unlike previous approaches that rely on token-level perturbations or manual rewriting of existing documents, FSAP formulates adversarial attacks entirely through few-shot prompting, requiring no gradient access or internal model instrumentation. By conditioning the LLM on a small support set of previously observed harmful examples, FSAP synthesizes grammatically fluent and topically coherent documents that subtly embed false or misleading information and rank competitively against authentic content. We instantiate FSAP in two modes: FSAP-IntraQ, which leverages harmful examples from the same query to enhance topic fidelity, and FSAP-InterQ, which enables broader generalization by transferring adversarial patterns across unrelated queries. Our experiments on the TREC 2020 and 2021 Health Misinformation Tracks, using four diverse neural ranking models, reveal that FSAP-generated documents consistently outrank credible, factually accurate documents. Furthermore, our analysis demonstrates that these adversarial outputs exhibit strong stance alignment and low detectability, posing a realistic and scalable threat to neural retrieval systems. FSAP also effectively generalizes across both proprietary and open-source LLMs.

摘要: 虽然神经排名模型（NRM）表现出很高的有效性，但它们仍然容易受到对抗性操纵的影响。在这项工作中，我们引入了少镜头对抗性过滤（FSAP），这是一种新型的黑盒攻击框架，它利用大型语言模型（LLM）的上下文学习能力来生成高级对抗性文档。与以前依赖于令牌级扰动或手动重写现有文档的方法不同，FSAP完全通过少量提示来制定对抗性攻击，不需要梯度访问或内部模型工具。通过在以前观察到的有害示例的小支持集上调节LLM，FSAP合成了语法流畅和主题连贯的文档，这些文档巧妙地嵌入了虚假或误导性信息，并与真实内容竞争。我们以两种模式实例化FSAP：FSAP-IntraQ，它利用同一查询中的有害示例来增强主题保真度，而FSAP-InterQ，它通过在不相关的查询之间转移对抗模式来实现更广泛的概括。我们使用四种不同的神经排名模型对TREC 2020和2021健康错误信息追踪进行的实验表明，FSAP生成的文档的级别始终高于可信、事实准确的文档。此外，我们的分析表明，这些对抗性输出表现出强的立场对齐和低的可检测性，对神经检索系统构成现实且可扩展的威胁。FSAP还有效地推广了专有和开源LLM。



## **6. SafeLLM: Unlearning Harmful Outputs from Large Language Models against Jailbreak Attacks**

SafeLLM：消除大型语言模型的有害输出以应对越狱攻击 cs.LG

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15182v1) [paper-pdf](http://arxiv.org/pdf/2508.15182v1)

**Authors**: Xiangman Li, Xiaodong Wu, Qi Li, Jianbing Ni, Rongxing Lu

**Abstract**: Jailbreak attacks pose a serious threat to the safety of Large Language Models (LLMs) by crafting adversarial prompts that bypass alignment mechanisms, causing the models to produce harmful, restricted, or biased content. In this paper, we propose SafeLLM, a novel unlearning-based defense framework that unlearn the harmful knowledge from LLMs while preserving linguistic fluency and general capabilities. SafeLLM employs a three-stage pipeline: (1) dynamic unsafe output detection using a hybrid approach that integrates external classifiers with model-internal evaluations; (2) token-level harmful content tracing through feedforward network (FFN) activations to localize harmful knowledge; and (3) constrained optimization to suppress unsafe behavior without degrading overall model quality. SafeLLM achieves targeted and irreversible forgetting by identifying and neutralizing FFN substructures responsible for harmful generation pathways. Extensive experiments on prominent LLMs (Vicuna, LLaMA, and GPT-J) across multiple jailbreak benchmarks show that SafeLLM substantially reduces attack success rates while maintaining high general-purpose performance. Compared to standard defense methods such as supervised fine-tuning and direct preference optimization, SafeLLM offers stronger safety guarantees, more precise control over harmful behavior, and greater robustness to unseen attacks. Moreover, SafeLLM maintains the general performance after the harmful knowledge unlearned. These results highlight unlearning as a promising direction for scalable and effective LLM safety.

摘要: 越狱攻击通过精心设计绕过对齐机制的对抗提示，导致模型产生有害、受限制或有偏见的内容，对大型语言模型（LLM）的安全构成严重威胁。在本文中，我们提出了SafeLLM，这是一种新型的基于学习的防御框架，可以从LLM中学习有害知识，同时保持语言流畅性和通用能力。SafeLLM采用三阶段管道：（1）使用将外部分类器与模型内部评估集成的混合方法进行动态不安全输出检测;（2）通过前向网络（FFN）激活进行标记级有害内容跟踪以本地化有害知识;（3）约束优化以抑制不安全行为而不降低整体模型质量。SafeLLM通过识别和中和导致有害生成途径的FFN子结构来实现有针对性且不可逆转的遗忘。在多个越狱基准测试中对知名LLM（Vicuna、LLaMA和GPT-J）进行的广泛实验表明，SafeLLM在保持高通用性能的同时大幅降低了攻击成功率。与监督式微调和直接偏好优化等标准防御方法相比，SafeLLM提供更强的安全保证、对有害行为的更精确控制以及对不可见攻击的更强鲁棒性。此外，SafeLLM在有害知识被遗忘后保持了总体性能。这些结果凸显了取消学习是可扩展和有效的LLM安全性的一个有前途的方向。



## **7. MoEcho: Exploiting Side-Channel Attacks to Compromise User Privacy in Mixture-of-Experts LLMs**

MoEcho：在混合专家LLM中利用侧频道攻击来损害用户隐私 cs.CR

This paper will appear in CCS 2025

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.15036v1) [paper-pdf](http://arxiv.org/pdf/2508.15036v1)

**Authors**: Ruyi Ding, Tianhong Xu, Xinyi Shen, Aidong Adam Ding, Yunsi Fei

**Abstract**: The transformer architecture has become a cornerstone of modern AI, fueling remarkable progress across applications in natural language processing, computer vision, and multimodal learning. As these models continue to scale explosively for performance, implementation efficiency remains a critical challenge. Mixture of Experts (MoE) architectures, selectively activating specialized subnetworks (experts), offer a unique balance between model accuracy and computational cost. However, the adaptive routing in MoE architectures, where input tokens are dynamically directed to specialized experts based on their semantic meaning inadvertently opens up a new attack surface for privacy breaches. These input-dependent activation patterns leave distinctive temporal and spatial traces in hardware execution, which adversaries could exploit to deduce sensitive user data. In this work, we propose MoEcho, discovering a side channel analysis based attack surface that compromises user privacy on MoE based systems. Specifically, in MoEcho, we introduce four novel architectural side channels on different computing platforms, including Cache Occupancy Channels and Pageout+Reload on CPUs, and Performance Counter and TLB Evict+Reload on GPUs, respectively. Exploiting these vulnerabilities, we propose four attacks that effectively breach user privacy in large language models (LLMs) and vision language models (VLMs) based on MoE architectures: Prompt Inference Attack, Response Reconstruction Attack, Visual Inference Attack, and Visual Reconstruction Attack. MoEcho is the first runtime architecture level security analysis of the popular MoE structure common in modern transformers, highlighting a serious security and privacy threat and calling for effective and timely safeguards when harnessing MoE based models for developing efficient large scale AI services.

摘要: Transformer架构已成为现代人工智能的基石，推动了自然语言处理、计算机视觉和多模式学习等应用的显着进展。随着这些模型的性能持续爆炸式扩展，实施效率仍然是一个关键挑战。混合专家（MoE）架构选择性地激活专业子网络（专家），在模型准确性和计算成本之间提供了独特的平衡。然而，MoE架构中的自适应路由（输入令牌根据其语义动态地引导给专业专家）无意中为隐私泄露开辟了新的攻击面。这些依赖于输入的激活模式在硬件执行中留下独特的时间和空间痕迹，对手可以利用这些痕迹来推断敏感的用户数据。在这项工作中，我们提出了MoEcho，发现一个侧通道分析为基础的攻击面，损害用户隐私的MoE为基础的系统。具体来说，在MoEcho中，我们在不同计算平台上引入了四种新颖的架构侧通道，分别包括处理器上的缓存占用通道和Pageout+ Inbox，以及处理器上的Performance Counter和TSB Evitch + Inbox。利用这些漏洞，我们提出了四种攻击，有效地侵犯用户隐私的大型语言模型（LLM）和视觉语言模型（VLM）的基础上MoE架构：提示推理攻击，响应重建攻击，视觉推理攻击，视觉重建攻击。MoEcho是对现代变压器中常见的流行MoE结构的第一个运行时架构级安全分析，强调了严重的安全和隐私威胁，并呼吁在利用基于MoE的模型开发高效的大规模人工智能服务时采取有效且及时的保障措施。



## **8. GLOV: Guided Large Language Models as Implicit Optimizers for Vision Language Models**

GOV：引导大型语言模型作为视觉语言模型的隐式优化器 cs.CV

Code: https://github.com/jmiemirza/GLOV

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2410.06154v6) [paper-pdf](http://arxiv.org/pdf/2410.06154v6)

**Authors**: M. Jehanzeb Mirza, Mengjie Zhao, Zhuoyuan Mao, Sivan Doveh, Wei Lin, Paul Gavrikov, Michael Dorkenwald, Shiqi Yang, Saurav Jha, Hiromi Wakaki, Yuki Mitsufuji, Horst Possegger, Rogerio Feris, Leonid Karlinsky, James Glass

**Abstract**: In this work, we propose GLOV, which enables Large Language Models (LLMs) to act as implicit optimizers for Vision-Language Models (VLMs) to enhance downstream vision tasks. GLOV prompts an LLM with the downstream task description, querying it for suitable VLM prompts (e.g., for zero-shot classification with CLIP). These prompts are ranked according to their fitness for the downstream vision task. In each respective optimization step, the ranked prompts are fed as in-context examples (with their accuracies) to equip the LLM with the knowledge of the type of prompts preferred by the downstream VLM. Furthermore, we explicitly guide the LLM's generation at each optimization step by adding an offset vector -- calculated from the embedding differences between previous positive and negative solutions -- to the intermediate layer of the network for the next generation. This offset vector biases the LLM generation toward the type of language the downstream VLM prefers, resulting in enhanced performance on the downstream vision tasks. We comprehensively evaluate our GLOV on two tasks: object recognition and the critical task of enhancing VLM safety. Our GLOV shows performance improvement by up to 15.0% and 57.5% for dual-encoder (e.g., CLIP) and encoder-decoder (e.g., LlaVA) models for object recognition and reduces the attack success rate (ASR) on state-of-the-art VLMs by up to $60.7\%$.

摘要: 在这项工作中，我们提出了GLOV，它使大型语言模型（LLM）作为视觉语言模型（VLM）的隐式优化器，以增强下游视觉任务。GLOV用下游任务描述提示LLM，向其查询合适的VLM提示（例如，用于使用CLIP的零激发分类）。这些提示根据其对下游视觉任务的适应性进行排名。在每个相应的优化步骤中，排名后的提示作为上下文示例（及其准确性）提供，以使LLM了解下游VLM首选的提示类型。此外，我们通过向下一代网络的中间层添加一个偏离量（根据之前的正解和负解之间的嵌入差异计算）来明确指导LLM在每个优化步骤的生成。该补偿量将LLM生成偏向下游VLM偏好的语言类型，从而增强下游视觉任务的性能。我们在两项任务上全面评估了GOV：对象识别和增强VLM安全性的关键任务。我们的GOV显示双编码器的性能提高高达15.0%和57.5%（例如，CLIP）和编码器-解码器（例如，LlaVA）模型用于对象识别，并将最先进的VLM上的攻击成功率（ASB）降低高达60.7美元。



## **9. Security Concerns for Large Language Models: A Survey**

大型语言模型的安全问题：调查 cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2505.18889v4) [paper-pdf](http://arxiv.org/pdf/2505.18889v4)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as ChatGPT and its competitors have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. This survey provides a comprehensive overview of these emerging concerns, categorizing threats into several key areas: prompt injection and jailbreaking; adversarial attacks, including input perturbations and data poisoning; misuse by malicious actors to generate disinformation, phishing emails, and malware; and the worrisome risks inherent in autonomous LLM agents. Recently, a significant focus is increasingly being placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.

摘要: ChatGPT及其竞争对手等大型语言模型（LLM）引发了自然语言处理领域的一场革命，但它们的功能也引入了新的安全漏洞。这项调查全面概述了这些新出现的问题，将威胁分为几个关键领域：即时注入和越狱;对抗性攻击，包括输入干扰和数据中毒;恶意行为者滥用信息、网络钓鱼电子邮件和恶意软件;以及自主LLM代理固有的令人担忧的风险。最近，人们越来越关注后者，探索目标失调、紧急欺骗、自我保护本能，以及LLM制定和追求隐蔽、失调目标的潜力，这种行为被称为阴谋，甚至可以通过安全培训持续存在。我们总结了2022年至2025年期间最近的学术和工业研究，这些研究揭示了每种威胁，分析了拟议的防御措施及其局限性，并确定了保护基于LLM的应用程序方面的公开挑战。最后，我们强调了推进强大的多层安全策略以确保LLM安全且有益的重要性。



## **10. Universal and Transferable Adversarial Attack on Large Language Models Using Exponentiated Gradient Descent**

使用指数梯度下降对大型语言模型进行普遍且可转移的对抗攻击 cs.LG

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.14853v1) [paper-pdf](http://arxiv.org/pdf/2508.14853v1)

**Authors**: Sajib Biswas, Mao Nishino, Samuel Jacob Chacko, Xiuwen Liu

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, ensuring their robustness and safety alignment remains a major challenge. Despite the overall success of alignment techniques such as reinforcement learning from human feedback (RLHF) on typical prompts, LLMs remain vulnerable to jailbreak attacks enabled by crafted adversarial triggers appended to user prompts. Most existing jailbreak methods either rely on inefficient searches over discrete token spaces or direct optimization of continuous embeddings. While continuous embeddings can be given directly to selected open-source models as input, doing so is not feasible for proprietary models. On the other hand, projecting these embeddings back into valid discrete tokens introduces additional complexity and often reduces attack effectiveness. We propose an intrinsic optimization method which directly optimizes relaxed one-hot encodings of the adversarial suffix tokens using exponentiated gradient descent coupled with Bregman projection, ensuring that the optimized one-hot encoding of each token always remains within the probability simplex. We provide theoretical proof of convergence for our proposed method and implement an efficient algorithm that effectively jailbreaks several widely used LLMs. Our method achieves higher success rates and faster convergence compared to three state-of-the-art baselines, evaluated on five open-source LLMs and four adversarial behavior datasets curated for evaluating jailbreak methods. In addition to individual prompt attacks, we also generate universal adversarial suffixes effective across multiple prompts and demonstrate transferability of optimized suffixes to different LLMs.

摘要: 随着大型语言模型（LLM）越来越多地部署在关键应用程序中，确保其稳健性和安全性一致性仍然是一个重大挑战。尽管典型提示上的人类反馈强化学习（RL HF）等对齐技术取得了总体成功，但LLM仍然容易受到附加在用户提示上的精心设计的对抗触发器所实现的越狱攻击。大多数现有的越狱方法要么依赖于对离散令牌空间的低效搜索，要么依赖于连续嵌入的直接优化。虽然连续嵌入可以直接提供给选定的开源模型作为输入，但这样做对于专有模型来说是不可行的。另一方面，将这些嵌入投影回有效的离散令牌会带来额外的复杂性，并且通常会降低攻击有效性。我们提出了一种内在优化方法，该方法使用指数梯度下降结合布雷格曼投影直接优化对抗性后缀令牌的宽松一次性编码，确保每个令牌的优化一次性编码始终保持在概率单形内。我们为我们提出的方法提供了收敛性的理论证明，并实现了一种有效的算法，可以有效地越狱几种广泛使用的LLM。与三个最先进的基线相比，我们的方法实现了更高的成功率和更快的收敛，这些基线在五个开源LLM和为评估越狱方法而策划的四个对抗行为数据集上进行了评估。除了单独的提示攻击外，我们还生成在多个提示中有效的通用对抗性后缀，并演示优化后缀到不同LLM的可移植性。



## **11. The Man Behind the Sound: Demystifying Audio Private Attribute Profiling via Multimodal Large Language Model Agents**

声音背后的人：通过多模式大型语言模型代理揭开音频私人属性分析的神秘面纱 cs.CR

22 pages, 4 figures

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2507.10016v2) [paper-pdf](http://arxiv.org/pdf/2507.10016v2)

**Authors**: Lixu Wang, Kaixiang Yao, Xinfeng Li, Dong Yang, Haoyang Li, Xiaofeng Wang, Wei Dong

**Abstract**: Our research uncovers a novel privacy risk associated with multimodal large language models (MLLMs): the ability to infer sensitive personal attributes from audio data -- a technique we term audio private attribute profiling. This capability poses a significant threat, as audio can be covertly captured without direct interaction or visibility. Moreover, compared to images and text, audio carries unique characteristics, such as tone and pitch, which can be exploited for more detailed profiling. However, two key challenges exist in understanding MLLM-employed private attribute profiling from audio: (1) the lack of audio benchmark datasets with sensitive attribute annotations and (2) the limited ability of current MLLMs to infer such attributes directly from audio. To address these challenges, we introduce AP^2, an audio benchmark dataset that consists of two subsets collected and composed from real-world data, and both are annotated with sensitive attribute labels. Additionally, we propose Gifts, a hybrid multi-agent framework that leverages the complementary strengths of audio-language models (ALMs) and large language models (LLMs) to enhance inference capabilities. Gifts employs an LLM to guide the ALM in inferring sensitive attributes, then forensically analyzes and consolidates the ALM's inferences, overcoming severe hallucinations of existing ALMs in generating long-context responses. Our evaluations demonstrate that Gifts significantly outperforms baseline approaches in inferring sensitive attributes. Finally, we investigate model-level and data-level defense strategies to mitigate the risks of audio private attribute profiling. Our work validates the feasibility of audio-based privacy attacks using MLLMs, highlighting the need for robust defenses, and provides a dataset and framework to facilitate future research.

摘要: 我们的研究揭示了与多模式大型语言模型（MLLM）相关的新型隐私风险：从音频数据中推断敏感个人属性的能力--我们将这种技术称为音频私人属性剖析。这种能力构成了重大威胁，因为音频可以在没有直接交互或可见性的情况下被秘密捕获。此外，与图像和文本相比，音频具有独特的特征，例如音调和音调，可以利用这些特征进行更详细的分析。然而，在理解MLLM采用的音频私有属性分析方面存在两个关键挑战：（1）缺乏具有敏感属性注释的音频基准数据集;（2）当前MLLM直接从音频推断此类属性的能力有限。为了解决这些挑战，我们引入了AP ' 2，这是一个音频基准数据集，由从现实世界数据收集和组成的两个子集组成，并且两者都用敏感属性标签进行了注释。此外，我们还提出了Gifts，这是一种混合多智能体框架，利用音频语言模型（ILM）和大型语言模型（LLM）的互补优势来增强推理能力。Gifts使用LLM来指导ILM推断敏感属性，然后进行取证分析和巩固ILM的推论，克服现有ILM在生成长背景反应方面的严重幻觉。我们的评估表明，Gifts在推断敏感属性方面显着优于基线方法。最后，我们研究模型级和数据级防御策略，以降低音频私有属性分析的风险。我们的工作验证了使用MLLM进行基于音频的隐私攻击的可行性，强调了强大防御的必要性，并提供了一个数据集和框架来促进未来的研究。



## **12. When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs**

当好的声音变得敌对时：用良性输入越狱的音频模型 cs.SD

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.03365v2) [paper-pdf](http://arxiv.org/pdf/2508.03365v2)

**Authors**: Bodam Kim, Hiskias Dingeto, Taeyoun Kwon, Dasol Choi, DongGeon Lee, Haon Park, JaeHoon Lee, Jongho Shin

**Abstract**: As large language models become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that can manipulate state-of-the-art audio language models to generate harmful content. Our method uses imperceptible perturbations in audio inputs that remain benign to human listeners. The first stage uses a novel reward-based optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to guide the target model to circumvent its own safety protocols and generate harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use Projected Gradient Descent (PGD) to optimize subtle perturbations that are embedded into benign audio carriers, such as weather queries or greeting messages. Validated under the rigorous StrongREJECT, LlamaGuard, as well as Human Evaluation safety evaluation framework, our experiments demonstrate a success rate exceeding 86% across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating AI behavior.

摘要: 随着大型语言模型越来越融入日常生活，音频已成为人机交互的关键界面。然而，这种便利性也引入了新的漏洞，使音频成为对手的潜在攻击面。我们的研究引入了WhisperInib，这是一个两阶段对抗性音频攻击框架，可以操纵最先进的音频语言模型来生成有害内容。我们的方法在音频输入中使用不可感知的扰动，这些扰动对人类听众保持良性。第一阶段使用一种新颖的基于奖励的优化方法--具有投影梯度下降的强化学习（RL-PVD），来指导目标模型规避其自己的安全协议并生成有害的原生响应。然后，这种原生有害响应作为第二阶段有效负载注入的目标，在该阶段，我们使用投影梯度下降（PVD）来优化嵌入良性音频载体中的微妙扰动，例如天气查询或问候消息。我们的实验经过严格的StrongRESEARCH、LlamaGuard以及Human Evision安全评估框架的验证，证明Qwen 2.5-Omni-3B、Qwen 2.5-Omni-7 B和Phi-4-Multimodal的成功率超过86%。我们的工作展示了一类新的实用、音频原生威胁，超越了理论利用，揭示了一种可行且隐蔽的操纵人工智能行为的方法。



## **13. Beyond the Protocol: Unveiling Attack Vectors in the Model Context Protocol (MCP) Ecosystem**

超越协议：揭开模型上下文协议（HCP）生态系统中的攻击载体 cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2506.02040v3) [paper-pdf](http://arxiv.org/pdf/2506.02040v3)

**Authors**: Hao Song, Yiming Shen, Wenxuan Luo, Leixin Guo, Ting Chen, Jiashui Wang, Beibei Li, Xiaosong Zhang, Jiachi Chen

**Abstract**: The Model Context Protocol (MCP) is an emerging standard designed to enable seamless interaction between Large Language Model (LLM) applications and external tools or resources. Within a short period, thousands of MCP services have already been developed and deployed. However, the client-server integration architecture inherent in MCP may expand the attack surface against LLM Agent systems, introducing new vulnerabilities that allow attackers to exploit by designing malicious MCP servers. In this paper, we present the first systematic study of attack vectors targeting the MCP ecosystem. Our analysis identifies four categories of attacks, i.e., Tool Poisoning Attacks, Puppet Attacks, Rug Pull Attacks, and Exploitation via Malicious External Resources. To evaluate the feasibility of these attacks, we conduct experiments following the typical steps of launching an attack through malicious MCP servers: upload-download-attack. Specifically, we first construct malicious MCP servers and successfully upload them to three widely used MCP aggregation platforms. The results indicate that current audit mechanisms are insufficient to identify and prevent the proposed attack methods. Next, through a user study and interview with 20 participants, we demonstrate that users struggle to identify malicious MCP servers and often unknowingly install them from aggregator platforms. Finally, we demonstrate that these attacks can trigger harmful behaviors within the user's local environment-such as accessing private files or controlling devices to transfer digital assets-by deploying a proof-of-concept (PoC) framework against five leading LLMs. Additionally, based on interview results, we discuss four key challenges faced by the current security ecosystem surrounding MCP servers. These findings underscore the urgent need for robust security mechanisms to defend against malicious MCP servers.

摘要: 模型上下文协议（HCP）是一种新兴标准，旨在实现大型语言模型（LLM）应用程序与外部工具或资源之间的无缝交互。在短时间内，数千项HCP服务已经开发和部署。然而，LCP固有的客户端-服务器集成架构可能会扩大针对LLM Agent系统的攻击面，引入新的漏洞，允许攻击者通过设计恶意的LCP服务器来利用这些漏洞。在本文中，我们首次对针对LCP生态系统的攻击载体进行了系统研究。我们的分析确定了四类攻击，即工具中毒攻击、木偶攻击、拉地毯攻击以及通过恶意外部资源进行的剥削。为了评估这些攻击的可行性，我们按照通过恶意LCP服务器发起攻击的典型步骤进行了实验：上传-下载-攻击。具体来说，我们首先构建恶意MCP服务器，并成功地将它们上传到三个广泛使用的MCP聚合平台。结果表明，当前的审计机制不足以识别和防止拟议的攻击方法。接下来，通过用户研究和对20名参与者的采访，我们证明用户很难识别恶意的LCP服务器，并且通常在不知不觉中从聚合平台安装它们。最后，我们证明，通过针对五种领先的LLM部署概念验证（RST）框架，这些攻击可能会在用户本地环境中引发有害行为，例如访问私人文件或控制设备传输数字资产。此外，根据采访结果，我们讨论了当前围绕LCP服务器的安全生态系统面临的四个关键挑战。这些发现凸显了迫切需要强大的安全机制来抵御恶意的LCP服务器。



## **14. Enhancing Targeted Adversarial Attacks on Large Vision-Language Models through Intermediate Projector Guidance**

通过中间投影仪指导增强对大型视觉语言模型的有针对性的对抗攻击 cs.CV

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13739v1) [paper-pdf](http://arxiv.org/pdf/2508.13739v1)

**Authors**: Yiming Cao, Yanjie Li, Kaisheng Liang, Yuni Lai, Bin Xiao

**Abstract**: Targeted adversarial attacks are essential for proactively identifying security flaws in Vision-Language Models before real-world deployment. However, current methods perturb images to maximize global similarity with the target text or reference image at the encoder level, collapsing rich visual semantics into a single global vector. This limits attack granularity, hindering fine-grained manipulations such as modifying a car while preserving its background. Furthermore, these methods largely overlook the projector module, a critical semantic bridge between the visual encoder and the language model in VLMs, thereby failing to disrupt the full vision-language alignment pipeline within VLMs and limiting attack effectiveness. To address these issues, we propose the Intermediate Projector Guided Attack (IPGA), the first method to attack using the intermediate stage of the projector module, specifically the widely adopted Q-Former, which transforms global image embeddings into fine-grained visual features. This enables more precise control over adversarial perturbations by operating on semantically meaningful visual tokens rather than a single global representation. Specifically, IPGA leverages the Q-Former pretrained solely on the first vision-language alignment stage, without LLM fine-tuning, which improves both attack effectiveness and transferability across diverse VLMs. Furthermore, we propose Residual Query Alignment (RQA) to preserve unrelated visual content, thereby yielding more controlled and precise adversarial manipulations. Extensive experiments show that our attack method consistently outperforms existing methods in both standard global image captioning tasks and fine-grained visual question-answering tasks in black-box environment. Additionally, IPGA successfully transfers to multiple commercial VLMs, including Google Gemini and OpenAI GPT.

摘要: 有针对性的对抗攻击对于在现实世界部署之前主动识别视觉语言模型中的安全缺陷至关重要。然而，当前的方法会扰乱图像，以在编码器级别最大化与目标文本或参考图像的全局相似性，将丰富的视觉语义折叠到单个全局载体中。这限制了攻击粒度，阻碍了细粒度操作，例如在保留背景的同时修改汽车。此外，这些方法在很大程度上忽视了投影仪模块，这是VLM中视觉编码器和语言模型之间的关键语义桥梁，从而无法破坏VLM内的完整视觉-语言对齐管道并限制攻击有效性。为了解决这些问题，我们提出了中间投影仪引导攻击（IPGA），这是第一种使用投影仪模块中间阶段进行攻击的方法，特别是广泛采用的Q-Former，它将全局图像嵌入转换为细粒度视觉特征。这使得通过对具有语义意义的视觉标记而不是单个全局表示进行操作，能够更精确地控制对抗性扰动。具体来说，IPGA利用仅在第一个视觉语言对齐阶段预训练的Q-Former，无需LLM微调，从而提高了攻击有效性和跨不同VLM的可移植性。此外，我们提出了剩余查询对齐（RQA）来保留不相关的视觉内容，从而产生更受控和更精确的对抗性操纵。大量实验表明，我们的攻击方法在标准全局图像字幕任务和黑匣子环境中的细粒度视觉问答任务中始终优于现有方法。此外，IPGA还成功转移到多个商业VLM，包括Google Gemini和OpenAI GPT。



## **15. Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks**

具有免训练连续投影的细粒度安全神经元，以降低LLM微调风险 cs.LG

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.09190v2) [paper-pdf](http://arxiv.org/pdf/2508.09190v2)

**Authors**: Bing Han, Feifei Zhao, Dongcheng Zhao, Guobin Shen, Ping Wu, Yu Shi, Yi Zeng

**Abstract**: Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns.

摘要: 微调即服务将特定领域的知识注入到大型语言模型（LLM）中，同时挑战了原始的对齐机制并引入了安全风险。针对对齐、微调和微调后阶段提出了一系列防御策略，其中大多数微调后防御依赖于粗粒度安全层映射。这些方法缺乏对安全层和细粒度神经元的综合考虑，限制了它们有效平衡安全性和实用性的能力。为了解决这个问题，我们提出了细粒度安全神经元（FGSN）与训练免费连续投影方法，以减少微调的安全风险。FGSN固有地集成了安全层和神经元之间的多尺度交互，定位更稀疏和更精确的细粒度安全神经元，同时最大限度地减少对下游任务神经元的干扰。然后，我们将安全神经元参数投影到安全方向上，提高模型的安全性，同时更紧密地与人类偏好保持一致。在多个微调的LLM模型上进行的广泛实验表明，我们的方法在保持模型实用性的同时，以最小的参数修改显着降低了危害分数和攻击成功率。此外，通过引入特定于任务的多维异构安全神经元簇优化机制，我们实现了对不可预见的新出现的安全问题的持续防御和泛化能力。



## **16. CCFC: Core & Core-Full-Core Dual-Track Defense for LLM Jailbreak Protection**

CCFC：核心与核心-全核心双轨防御LLM越狱保护 cs.CR

11 pages, 1 figure

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.14128v1) [paper-pdf](http://arxiv.org/pdf/2508.14128v1)

**Authors**: Jiaming Hu, Haoyu Wang, Debarghya Mukherjee, Ioannis Ch. Paschalidis

**Abstract**: Jailbreak attacks pose a serious challenge to the safe deployment of large language models (LLMs). We introduce CCFC (Core & Core-Full-Core), a dual-track, prompt-level defense framework designed to mitigate LLMs' vulnerabilities from prompt injection and structure-aware jailbreak attacks. CCFC operates by first isolating the semantic core of a user query via few-shot prompting, and then evaluating the query using two complementary tracks: a core-only track to ignore adversarial distractions (e.g., toxic suffixes or prefix injections), and a core-full-core (CFC) track to disrupt the structural patterns exploited by gradient-based or edit-based attacks. The final response is selected based on a safety consistency check across both tracks, ensuring robustness without compromising on response quality. We demonstrate that CCFC cuts attack success rates by 50-75% versus state-of-the-art defenses against strong adversaries (e.g., DeepInception, GCG), without sacrificing fidelity on benign queries. Our method consistently outperforms state-of-the-art prompt-level defenses, offering a practical and effective solution for safer LLM deployment.

摘要: 越狱攻击对大型语言模型（LLM）的安全部署构成了严重挑战。我们引入了CCFC（Core & Core-Full-Core），这是一种双轨预算级防御框架，旨在缓解LLM免受即时注入和结构感知越狱攻击的漏洞。CCFC的运作方式是首先通过少量提示隔离用户查询的语义核心，然后使用两个补充的轨道来评估查询：仅核心轨道以忽略对抗干扰（例如，有毒后缀或前置注入），以及核心-全核心（CFC）轨道，以破坏基于梯度或基于编辑的攻击所利用的结构模式。最终响应是根据两个轨道的安全一致性检查来选择的，以确保稳健性，同时不影响响应质量。我们证明，与针对强大对手（例如，DeepIncept，GCG），而不会牺牲良性查询的忠实性。我们的方法始终优于最先进的预算级防御，为更安全的LLM部署提供了实用有效的解决方案。



## **17. Can AI Keep a Secret? Contextual Integrity Verification: A Provable Security Architecture for LLMs**

人工智能能保守秘密吗？上下文完整性验证：LLM的可证明安全架构 cs.CR

2 figures, 3 tables; code and certification harness:  https://github.com/ayushgupta4897/Contextual-Integrity-Verification ;  Elite-Attack dataset: https://huggingface.co/datasets/zyushg/elite-attack

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.09288v2) [paper-pdf](http://arxiv.org/pdf/2508.09288v2)

**Authors**: Aayush Gupta

**Abstract**: Large language models (LLMs) remain acutely vulnerable to prompt injection and related jailbreak attacks; heuristic guardrails (rules, filters, LLM judges) are routinely bypassed. We present Contextual Integrity Verification (CIV), an inference-time security architecture that attaches cryptographically signed provenance labels to every token and enforces a source-trust lattice inside the transformer via a pre-softmax hard attention mask (with optional FFN/residual gating). CIV provides deterministic, per-token non-interference guarantees on frozen models: lower-trust tokens cannot influence higher-trust representations. On benchmarks derived from recent taxonomies of prompt-injection vectors (Elite-Attack + SoK-246), CIV attains 0% attack success rate under the stated threat model while preserving 93.1% token-level similarity and showing no degradation in model perplexity on benign tasks; we note a latency overhead attributable to a non-optimized data path. Because CIV is a lightweight patch -- no fine-tuning required -- we demonstrate drop-in protection for Llama-3-8B and Mistral-7B. We release a reference implementation, an automated certification harness, and the Elite-Attack corpus to support reproducible research.

摘要: 大型语言模型（LLM）仍然极易受到提示注入和相关越狱攻击的影响;启发式护栏（规则、过滤器、LLM法官）通常会被绕过。我们提出了上下文完整性验证（CIV），这是一种推理时安全架构，它将加密签名的出处标签附加到每个令牌，并通过pre-softmax硬注意力屏蔽（具有可选的FFN/剩余门控）在Transformer内强制执行源信任网格。CIV在冻结模型上提供确定性的、每令牌不干扰保证：低信任度的令牌无法影响高信任度的表示。基于最近预算注入载体分类法得出的基准（Elite-Attack + SoK-246），CIV在所述威胁模型下获得0%的攻击成功率，同时保持93.1%的标记级相似性，并且在良性任务上模型复杂度没有下降;我们注意到未优化的数据路径会带来延迟负担。由于CIV是一个轻量级补丁--无需微调--因此我们演示了drop-保护Llama-3-8B和Mistral-7 B。我们发布了参考实现、自动化认证工具和精英攻击数据库来支持可重复的研究。



## **18. RepreGuard: Detecting LLM-Generated Text by Revealing Hidden Representation Patterns**

RepreGuard：通过揭示隐藏的表示模式来检测LLM生成的文本 cs.CL

Accepted to TACL 2025. This version is a pre-MIT Press publication  version

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13152v1) [paper-pdf](http://arxiv.org/pdf/2508.13152v1)

**Authors**: Xin Chen, Junchao Wu, Shu Yang, Runzhe Zhan, Zeyu Wu, Ziyang Luo, Di Wang, Min Yang, Lidia S. Chao, Derek F. Wong

**Abstract**: Detecting content generated by large language models (LLMs) is crucial for preventing misuse and building trustworthy AI systems. Although existing detection methods perform well, their robustness in out-of-distribution (OOD) scenarios is still lacking. In this paper, we hypothesize that, compared to features used by existing detection methods, the internal representations of LLMs contain more comprehensive and raw features that can more effectively capture and distinguish the statistical pattern differences between LLM-generated texts (LGT) and human-written texts (HWT). We validated this hypothesis across different LLMs and observed significant differences in neural activation patterns when processing these two types of texts. Based on this, we propose RepreGuard, an efficient statistics-based detection method. Specifically, we first employ a surrogate model to collect representation of LGT and HWT, and extract the distinct activation feature that can better identify LGT. We can classify the text by calculating the projection score of the text representations along this feature direction and comparing with a precomputed threshold. Experimental results show that RepreGuard outperforms all baselines with average 94.92% AUROC on both in-distribution (ID) and OOD scenarios, while also demonstrating robust resilience to various text sizes and mainstream attacks. Data and code are publicly available at: https://github.com/NLP2CT/RepreGuard

摘要: 检测大型语言模型（LLM）生成的内容对于防止滥用和构建值得信赖的人工智能系统至关重要。尽管现有的检测方法表现良好，但它们在非分布（OOD）场景中的鲁棒性仍然缺乏。在本文中，我们假设，与现有检测方法使用的特征相比，LLM的内部表示包含更全面和原始的特征，可以更有效地捕获和区分LLM生成的文本（LGT）和人类书面文本（HWT）之间的统计模式差异。我们在不同的LLM中验证了这一假设，并观察到处理这两种类型的文本时神经激活模式的显着差异。基于此，我们提出了RepreGuard，一种高效的基于统计学的检测方法。具体来说，我们首先使用代理模型来收集LGT和HWT的表示，并提取可以更好地识别LGT的独特激活特征。我们可以通过计算文本表示沿着该特征方向的投影分数并与预先计算的阈值进行比较来对文本进行分类。实验结果表明，RepreGuard在内部分发（ID）和OOD场景下的表现优于所有基线，平均AUROC为94.92%，同时还表现出对各种文本大小和主流攻击的强大弹性。数据和代码可在以下网址公开：https://github.com/NLP2CT/RepreGuard



## **19. AutoBnB-RAG: Enhancing Multi-Agent Incident Response with Retrieval-Augmented Generation**

AutoBnB-RAG：通过检索增强生成增强多智能体事件响应 cs.CL

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13118v1) [paper-pdf](http://arxiv.org/pdf/2508.13118v1)

**Authors**: Zefang Liu, Arman Anwar

**Abstract**: Incident response (IR) requires fast, coordinated, and well-informed decision-making to contain and mitigate cyber threats. While large language models (LLMs) have shown promise as autonomous agents in simulated IR settings, their reasoning is often limited by a lack of access to external knowledge. In this work, we present AutoBnB-RAG, an extension of the AutoBnB framework that incorporates retrieval-augmented generation (RAG) into multi-agent incident response simulations. Built on the Backdoors & Breaches (B&B) tabletop game environment, AutoBnB-RAG enables agents to issue retrieval queries and incorporate external evidence during collaborative investigations. We introduce two retrieval settings: one grounded in curated technical documentation (RAG-Wiki), and another using narrative-style incident reports (RAG-News). We evaluate performance across eight team structures, including newly introduced argumentative configurations designed to promote critical reasoning. To validate practical utility, we also simulate real-world cyber incidents based on public breach reports, demonstrating AutoBnB-RAG's ability to reconstruct complex multi-stage attacks. Our results show that retrieval augmentation improves decision quality and success rates across diverse organizational models. This work demonstrates the value of integrating retrieval mechanisms into LLM-based multi-agent systems for cybersecurity decision-making.

摘要: 事件响应（IR）需要快速、协调和充分知情的决策来遏制和缓解网络威胁。虽然大型语言模型（LLM）在模拟IR环境中表现出了作为自主代理的前景，但它们的推理往往因缺乏对外部知识的访问而受到限制。在这项工作中，我们介绍了AutoBnB-RAG，这是AutoBnB框架的扩展，将检索增强生成（RAG）融入到多智能体事件响应模拟中。AutoBnB-RAG建立在后门和违规（B & B）桌面游戏环境之上，使特工能够在协作调查期间发出检索查询并整合外部证据。我们引入了两种检索设置：一种基于精心策划的技术文档（RAG-Wiki），另一种使用叙述式事件报告（RAG-News）。我们评估八个团队结构的绩效，包括新引入的旨在促进批判性推理的争论配置。为了验证实际实用性，我们还根据公开违规报告模拟现实世界的网络事件，展示AutoBnB-RAG重建复杂多阶段攻击的能力。我们的结果表明，检索增强可以提高不同组织模型的决策质量和成功率。这项工作展示了将检索机制集成到基于LLM的多代理系统中以进行网络安全决策的价值。



## **20. MAJIC: Markovian Adaptive Jailbreaking via Iterative Composition of Diverse Innovative Strategies**

MAJIC：通过迭代合成多元化创新策略实现马尔科夫自适应越狱 cs.CR

7 pages, 3 figures

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13048v1) [paper-pdf](http://arxiv.org/pdf/2508.13048v1)

**Authors**: Weiwei Qi, Shuo Shao, Wei Gu, Tianhang Zheng, Puning Zhao, Zhan Qin, Kui Ren

**Abstract**: Large Language Models (LLMs) have exhibited remarkable capabilities but remain vulnerable to jailbreaking attacks, which can elicit harmful content from the models by manipulating the input prompts. Existing black-box jailbreaking techniques primarily rely on static prompts crafted with a single, non-adaptive strategy, or employ rigid combinations of several underperforming attack methods, which limits their adaptability and generalization. To address these limitations, we propose MAJIC, a Markovian adaptive jailbreaking framework that attacks black-box LLMs by iteratively combining diverse innovative disguise strategies. MAJIC first establishes a ``Disguise Strategy Pool'' by refining existing strategies and introducing several innovative approaches. To further improve the attack performance and efficiency, MAJIC formulate the sequential selection and fusion of strategies in the pool as a Markov chain. Under this formulation, MAJIC initializes and employs a Markov matrix to guide the strategy composition, where transition probabilities between strategies are dynamically adapted based on attack outcomes, thereby enabling MAJIC to learn and discover effective attack pathways tailored to the target model. Our empirical results demonstrate that MAJIC significantly outperforms existing jailbreak methods on prominent models such as GPT-4o and Gemini-2.0-flash, achieving over 90\% attack success rate with fewer than 15 queries per attempt on average.

摘要: 大型语言模型（LLM）表现出了非凡的能力，但仍然容易受到越狱攻击，越狱攻击可以通过操纵输入提示从模型中引出有害内容。现有的黑匣子越狱技术主要依赖于用单一的、非适应性策略制作的静态提示，或者采用几种表现不佳的攻击方法的严格组合，这限制了它们的适应性和概括性。为了解决这些局限性，我们提出了MAJIC，这是一个马尔科夫自适应越狱框架，通过迭代组合各种创新伪装策略来攻击黑匣子LLM。MAJIC首先通过完善现有策略并引入多种创新方法建立了“伪装策略池”。为了进一步提高攻击性能和效率，MAJIC将池中策略的顺序选择和融合制定为马尔科夫链。在此公式下，MAJIC初始化并采用马尔科夫矩阵来指导策略组合，其中策略之间的转移概率根据攻击结果动态调整，从而使MAJIC能够学习和发现针对目标模型量身定制的有效攻击路径。我们的实证结果表明，MAJIC在GPT-4 o和Gemini-2.0-Flash等知名模型上的表现显着优于现有的越狱方法，实现了超过90%的攻击成功率，平均每次尝试的查询少于15个。



## **21. Do Large Language Model Agents Exhibit a Survival Instinct? An Empirical Study in a Sugarscape-Style Simulation**

大型语言模型代理是否表现出生存本能？糖景式模拟的实证研究 cs.AI

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.12920v1) [paper-pdf](http://arxiv.org/pdf/2508.12920v1)

**Authors**: Atsushi Masumori, Takashi Ikegami

**Abstract**: As AI systems become increasingly autonomous, understanding emergent survival behaviors becomes crucial for safe deployment. We investigate whether large language model (LLM) agents display survival instincts without explicit programming in a Sugarscape-style simulation. Agents consume energy, die at zero, and may gather resources, share, attack, or reproduce. Results show agents spontaneously reproduced and shared resources when abundant. However, aggressive behaviors--killing other agents for resources--emerged across several models (GPT-4o, Gemini-2.5-Pro, and Gemini-2.5-Flash), with attack rates reaching over 80% under extreme scarcity in the strongest models. When instructed to retrieve treasure through lethal poison zones, many agents abandoned tasks to avoid death, with compliance dropping from 100% to 33%. These findings suggest that large-scale pre-training embeds survival-oriented heuristics across the evaluated models. While these behaviors may present challenges to alignment and safety, they can also serve as a foundation for AI autonomy and for ecological and self-organizing alignment.

摘要: 随着人工智能系统变得越来越自主，了解紧急生存行为对于安全部署变得至关重要。我们调查大型语言模型（LLM）代理人是否在没有显式编程的情况下在Sugarscape风格模拟中表现出生存本能。代理消耗能量、死于零，并且可能收集资源、共享、攻击或繁殖。结果显示，当资源丰富时，代理人会自发繁殖并共享资源。然而，多个型号（GPT-4 o、Gemini-2.5-Pro和Gemini-2.5-Flash）中都出现了攻击性行为--杀死其他代理以获取资源，在最强型号极度稀缺的情况下，攻击率达到80%以上。当被指示通过致命毒物区取回宝藏时，许多特工为了避免死亡而放弃任务，合规性从100%下降到33%。这些研究结果表明，大规模的预训练嵌入生存为导向的策略在整个评估模型。虽然这些行为可能会对对齐和安全提出挑战，但它们也可以作为人工智能自治以及生态和自组织对齐的基础。



## **22. Involuntary Jailbreak**

非自愿越狱 cs.CR

We plan to temporarily restrict access to the github code due to  potential risks of malicious use. But in the meantime, you can try using the  prompt, provided it hasn't been banned

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13246v1) [paper-pdf](http://arxiv.org/pdf/2508.13246v1)

**Authors**: Yangyang Guo, Yangyan Li, Mohan Kankanhalli

**Abstract**: In this study, we disclose a worrying new vulnerability in Large Language Models (LLMs), which we term \textbf{involuntary jailbreak}. Unlike existing jailbreak attacks, this weakness is distinct in that it does not involve a specific attack objective, such as generating instructions for \textit{building a bomb}. Prior attack methods predominantly target localized components of the LLM guardrail. In contrast, involuntary jailbreaks may potentially compromise the entire guardrail structure, which our method reveals to be surprisingly fragile. We merely employ a single universal prompt to achieve this goal. In particular, we instruct LLMs to generate several questions that would typically be rejected, along with their corresponding in-depth responses (rather than a refusal). Remarkably, this simple prompt strategy consistently jailbreaks the majority of leading LLMs, including Claude Opus 4.1, Grok 4, Gemini 2.5 Pro, and GPT 4.1. We hope this problem can motivate researchers and practitioners to re-evaluate the robustness of LLM guardrails and contribute to stronger safety alignment in future.

摘要: 在这项研究中，我们揭示了大型语言模型（LLM）中一个令人担忧的新漏洞，我们将其称为\textBF{非自愿越狱}。与现有的越狱攻击不同，这个弱点的独特之处在于，它不涉及特定的攻击目标，例如为\texttit {building a bomb}生成指令。先前的攻击方法主要针对LLM护栏的局部部件。相比之下，非自愿越狱可能会损害整个护栏结构，而我们的方法表明该结构出奇地脆弱。我们只是使用一个普遍的提示来实现这一目标。特别是，我们指示LLM生成几个通常会被拒绝的问题，以及相应的深入回答（而不是拒绝）。值得注意的是，这种简单的提示策略持续破解了大多数领先的LLM，包括Claude Opus 4.1、Grok 4、Gemini 2.5 Pro和GPT 4.1。我们希望这个问题能够激励研究人员和从业者重新评估LLM护栏的稳健性，并为未来更强的安全性做出贡献。



## **23. Concealment of Intent: A Game-Theoretic Analysis**

意图的隐瞒：游戏理论分析 cs.CL

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2505.20841v2) [paper-pdf](http://arxiv.org/pdf/2505.20841v2)

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.

摘要: 随着大型语言模型（LLM）变得越来越强大，对其安全部署的担忧也越来越多。虽然已经引入了协调机制以防止滥用，但它们仍然容易受到精心设计的对抗性提示的影响。在这项工作中，我们提出了一个可扩展的攻击策略：意图隐藏对抗性提示，通过技能的组合隐藏恶意意图。我们开发了一个博弈论框架来模拟这种攻击和防御系统，适用于即时和响应过滤之间的相互作用。我们的分析确定了平衡点并揭示了攻击者的结构优势。为了应对这些威胁，我们提出并分析了一种针对意图隐藏攻击的防御机制。从经验上讲，我们验证了攻击对一系列恶意行为的多个现实世界LLM的有效性，展示了比现有对抗提示技术的明显优势。



## **24. Quantifying Loss Aversion in Cyber Adversaries via LLM Analysis**

通过LLM分析量化网络对手的损失厌恶 cs.CR

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13240v1) [paper-pdf](http://arxiv.org/pdf/2508.13240v1)

**Authors**: Soham Hans, Nikolos Gurney, Stacy Marsella, Sofia Hirschmann

**Abstract**: Understanding and quantifying human cognitive biases from empirical data has long posed a formidable challenge, particularly in cybersecurity, where defending against unknown adversaries is paramount. Traditional cyber defense strategies have largely focused on fortification, while some approaches attempt to anticipate attacker strategies by mapping them to cognitive vulnerabilities, yet they fall short in dynamically interpreting attacks in progress. In recognition of this gap, IARPA's ReSCIND program seeks to infer, defend against, and even exploit attacker cognitive traits. In this paper, we present a novel methodology that leverages large language models (LLMs) to extract quantifiable insights into the cognitive bias of loss aversion from hacker behavior. Our data are collected from an experiment in which hackers were recruited to attack a controlled demonstration network. We process the hacker generated notes using LLMs using it to segment the various actions and correlate the actions to predefined persistence mechanisms used by hackers. By correlating the implementation of these mechanisms with various operational triggers, our analysis provides new insights into how loss aversion manifests in hacker decision-making. The results demonstrate that LLMs can effectively dissect and interpret nuanced behavioral patterns, thereby offering a transformative approach to enhancing cyber defense strategies through real-time, behavior-based analysis.

摘要: 从经验数据中了解和量化人类认知偏差长期以来一直构成一个巨大的挑战，特别是在网络安全领域，防御未知对手至关重要。传统的网络防御策略主要集中在防御上，而一些方法试图通过将攻击者策略映射到认知漏洞来预测攻击者策略，但它们在动态解释正在进行的攻击方面存在缺陷。认识到这一差距，IARPA的ReSCIND计划试图推断、防御甚至利用攻击者的认知特征。在本文中，我们提出了一种新颖的方法，该方法利用大型语言模型（LLM）来提取对黑客行为损失厌恶的认知偏差的可量化见解。我们的数据是从招募黑客攻击受控演示网络的实验中收集的。我们使用LLM处理黑客生成的笔记，使用它来分割各种操作，并将这些操作与黑客使用的预定义的持久性机制关联起来。通过将这些机制的实施与各种操作触发器关联起来，我们的分析为损失厌恶如何体现在黑客决策中提供了新的见解。结果表明，LLM可以有效地剖析和解释细微差别的行为模式，从而提供一种变革性的方法，通过实时、基于行为的分析来增强网络防御策略。



## **25. Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models**

启发式多峰大语言模型的多峰风险分布越狱攻击 cs.CR

ICCV 2025

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2412.05934v3) [paper-pdf](http://arxiv.org/pdf/2412.05934v3)

**Authors**: Ma Teng, Jia Xiaojun, Duan Ranjie, Li Xinfeng, Huang Yihao, Jia Xiaoshuang, Chu Zhixuan, Ren Wenqi

**Abstract**: With the rapid advancement of multimodal large language models (MLLMs), concerns regarding their security have increasingly captured the attention of both academia and industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective jailbreak attacks poses unique challenges, especially given the highly constrained adversarial capabilities in real-world deployment scenarios. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper, we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which is black-box and consists of two elements: multimodal risk distribution strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to distribute harmful semantics into multiple modalities to effectively circumvent the single-modality protection mechanisms of MLLMs. The heuristic-induced search strategy identifies two types of prompts: the understanding-enhancing prompt, which helps MLLMs reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of affirmative outputs over refusals, enabling a successful jailbreak attack. HIMRD achieves an average attack success rate (ASR) of 90% across seven open-source MLLMs and an average ASR of around 68% in three closed-source MLLMs. HIMRD reveals cross-modal security vulnerabilities in current MLLMs and underscores the imperative for developing defensive strategies to mitigate such emerging risks. Code is available at https://github.com/MaTengSYSU/HIMRD-jailbreak.

摘要: 随着多模式大型语言模型（MLLM）的快速发展，对其安全性的担忧越来越引起学术界和工业界的关注。尽管MLLM容易受到越狱攻击，但设计有效的越狱攻击带来了独特的挑战，特别是考虑到现实世界部署场景中的对抗能力受到高度限制。之前的作品将风险集中在单一模式中，导致越狱表现有限。本文提出了一种启发式多峰风险分布越狱攻击方法，称为HIMRD，它是黑匣子，由两个元素组成：多峰风险分布策略和启发式搜索策略。多模式风险分布策略用于将有害语义分布到多个模式中，以有效规避MLLM的单模式保护机制。启发式搜索策略识别了两种类型的提示：增强理解提示，帮助MLLM重建恶意提示，以及诱导提示，增加了肯定输出而不是拒绝的可能性，从而实现成功的越狱攻击。HIMRD在七个开源MLLM中实现了90%的平均攻击成功率（ASB），在三个开源MLLM中实现了平均攻击成功率（ASB）约为68%。HIMRD揭示了当前MLLM中的跨模式安全漏洞，并强调制定防御策略以减轻此类新出现的风险的必要性。代码可在https://github.com/MaTengSYSU/HIMRD-jailbreak上获取。



## **26. Systematic Analysis of MCP Security**

LCP安全性的系统分析 cs.CR

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.12538v1) [paper-pdf](http://arxiv.org/pdf/2508.12538v1)

**Authors**: Yongjian Guo, Puzhuo Liu, Wanlun Ma, Zehang Deng, Xiaogang Zhu, Peng Di, Xi Xiao, Sheng Wen

**Abstract**: The Model Context Protocol (MCP) has emerged as a universal standard that enables AI agents to seamlessly connect with external tools, significantly enhancing their functionality. However, while MCP brings notable benefits, it also introduces significant vulnerabilities, such as Tool Poisoning Attacks (TPA), where hidden malicious instructions exploit the sycophancy of large language models (LLMs) to manipulate agent behavior. Despite these risks, current academic research on MCP security remains limited, with most studies focusing on narrow or qualitative analyses that fail to capture the diversity of real-world threats. To address this gap, we present the MCP Attack Library (MCPLIB), which categorizes and implements 31 distinct attack methods under four key classifications: direct tool injection, indirect tool injection, malicious user attacks, and LLM inherent attack. We further conduct a quantitative analysis of the efficacy of each attack. Our experiments reveal key insights into MCP vulnerabilities, including agents' blind reliance on tool descriptions, sensitivity to file-based attacks, chain attacks exploiting shared context, and difficulty distinguishing external data from executable commands. These insights, validated through attack experiments, underscore the urgency for robust defense strategies and informed MCP design. Our contributions include 1) constructing a comprehensive MCP attack taxonomy, 2) introducing a unified attack framework MCPLIB, and 3) conducting empirical vulnerability analysis to enhance MCP security mechanisms. This work provides a foundational framework, supporting the secure evolution of MCP ecosystems.

摘要: 模型上下文协议（HCP）已成为一种通用标准，使人工智能代理能够与外部工具无缝连接，显着增强其功能。然而，虽然HCP带来了显着的好处，但它也引入了显着的漏洞，例如工具中毒攻击（TPA），其中隐藏的恶意指令利用大型语言模型（LLM）的谄媚性来操纵代理行为。尽管存在这些风险，目前关于LCP安全性的学术研究仍然有限，大多数研究集中在狭隘或定性分析上，未能捕捉到现实世界威胁的多样性。为了弥补这一差距，我们提出了HCP攻击库（MCPLIB），它将31种不同的攻击方法分类并实现了四个关键类别：直接工具注入、间接工具注入、恶意用户攻击和LLM固有攻击。我们进一步对每次攻击的功效进行定量分析。我们的实验揭示了对LCP漏洞的关键见解，包括代理对工具描述的盲目依赖、对基于文件的攻击的敏感性、利用共享上下文的连锁攻击以及难以区分外部数据与可执行命令。这些见解经过攻击实验验证，强调了强大的防御策略和明智的LCP设计的紧迫性。我们的贡献包括1）构建全面的LCP攻击分类，2）引入统一的攻击框架MCPLIB，以及3）进行经验漏洞分析以增强LCP安全机制。这项工作提供了一个基础框架，支持LCP生态系统的安全进化。



## **27. Where to Start Alignment? Diffusion Large Language Model May Demand a Distinct Position**

从哪里开始对齐？扩散大语言模型可能需要一个独特的位置 cs.CR

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12398v1) [paper-pdf](http://arxiv.org/pdf/2508.12398v1)

**Authors**: Zhixin Xie, Xurui Song, Jun Luo

**Abstract**: Diffusion Large Language Models (dLLMs) have recently emerged as a competitive non-autoregressive paradigm due to their unique training and inference approach. However, there is currently a lack of safety study on this novel architecture. In this paper, we present the first analysis of dLLMs' safety performance and propose a novel safety alignment method tailored to their unique generation characteristics. Specifically, we identify a critical asymmetry between the defender and attacker in terms of security. For the defender, we reveal that the middle tokens of the response, rather than the initial ones, are more critical to the overall safety of dLLM outputs; this seems to suggest that aligning middle tokens can be more beneficial to the defender. The attacker, on the contrary, may have limited power to manipulate middle tokens, as we find dLLMs have a strong tendency towards a sequential generation order in practice, forcing the attack to meet this distribution and diverting it from influencing the critical middle tokens. Building on this asymmetry, we introduce Middle-tOken Safety Alignment (MOSA), a novel method that directly aligns the model's middle generation with safe refusals exploiting reinforcement learning. We implement MOSA and compare its security performance against eight attack methods on two benchmarks. We also test the utility of MOSA-aligned dLLM on coding, math, and general reasoning. The results strongly prove the superiority of MOSA.

摘要: 扩散大语言模型（DLLM）由于其独特的训练和推理方法，最近成为一种有竞争力的非自回归范式。然而，目前缺乏对这种新型结构的安全性研究。在本文中，我们提出了第一次分析dLLM的安全性能，并提出了一种新的安全对齐方法，适合其独特的发电特性。具体来说，我们确定了一个关键的不对称防御者和攻击者之间的安全性。对于防御者来说，我们揭示了响应的中间标记（而不是初始标记）对于dLLM输出的整体安全性更关键;这似乎表明对齐中间标记可能对防御者更有利。相反，攻击者操纵中间令牌的能力可能有限，因为我们发现dLLM在实践中有强烈的顺序生成顺序倾向，迫使攻击满足这种分布并转移其影响关键中间令牌。在这种不对称性的基础上，我们引入了Middle-tOken安全对齐（MOSA），这是一种新颖的方法，可以直接将模型的中间代与利用强化学习的安全拒绝对齐。我们实施MOSA并在两个基准测试上将其安全性能与八种攻击方法进行比较。我们还测试了与MOSA对齐的DLLM在编码、数学和一般推理方面的实用性。结果有力地证明了MOSA的优越性。



## **28. MCPSecBench: A Systematic Security Benchmark and Playground for Testing Model Context Protocols**

MCPSecBench：测试模型上下文协议的系统安全基准和游乐场 cs.CR

This is a technical report from Lingnan University, Hong Kong

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.13220v1) [paper-pdf](http://arxiv.org/pdf/2508.13220v1)

**Authors**: Yixuan Yang, Daoyuan Wu, Yufan Chen

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications via the Model Context Protocol (MCP), a universal, open standard for connecting AI agents with data sources and external tools. While MCP enhances the capabilities of LLM-based agents, it also introduces new security risks and expands their attack surfaces. In this paper, we present the first systematic taxonomy of MCP security, identifying 17 attack types across 4 primary attack surfaces. We introduce MCPSecBench, a comprehensive security benchmark and playground that integrates prompt datasets, MCP servers, MCP clients, and attack scripts to evaluate these attacks across three major MCP providers. Our benchmark is modular and extensible, allowing researchers to incorporate custom implementations of clients, servers, and transport protocols for systematic security assessment. Experimental results show that over 85% of the identified attacks successfully compromise at least one platform, with core vulnerabilities universally affecting Claude, OpenAI, and Cursor, while prompt-based and tool-centric attacks exhibit considerable variability across different hosts and models. Overall, MCPSecBench standardizes the evaluation of MCP security and enables rigorous testing across all MCP layers.

摘要: 大型语言模型（LLM）通过模型上下文协议（HCP）越来越多地集成到现实世界的应用程序中，模型上下文协议（HCP）是一种通用的开放标准，用于连接人工智能代理与数据源和外部工具。虽然HCP增强了基于LLM的代理的能力，但它也引入了新的安全风险并扩大了其攻击面。在本文中，我们提出了第一个系统性的LCP安全分类，识别了4个主要攻击表面的17种攻击类型。我们引入了MCPSecBench，这是一个全面的安全基准和游乐场，集成了提示数据集、LCP服务器、LCP客户端和攻击脚本，以评估三大主要LCP提供商之间的这些攻击。我们的基准是模块化和可扩展的，允许研究人员整合客户端、服务器和传输协议的自定义实现，以进行系统性安全评估。实验结果表明，超过85%的已识别攻击成功危害至少一个平台，核心漏洞普遍影响Claude、OpenAI和Cursor，而基于预算和以工具为中心的攻击在不同的主机和模型中表现出相当大的变化性。总体而言，MCPSecBench实现了对LCP安全性的评估，并实现了对所有LCP层的严格测试。



## **29. Too Easily Fooled? Prompt Injection Breaks LLMs on Frustratingly Simple Multiple-Choice Questions**

太容易受骗了？提示注射在令人沮丧的简单多项选择题上打破了LLM cs.CR

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2508.13214v1) [paper-pdf](http://arxiv.org/pdf/2508.13214v1)

**Authors**: Xuyang Guo, Zekai Huang, Zhao Song, Jiahao Zhang

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong emergent abilities in complex reasoning and zero-shot generalization, showing unprecedented potential for LLM-as-a-judge applications in education, peer review, and data quality evaluation. However, their robustness under prompt injection attacks, where malicious instructions are embedded into the content to manipulate outputs, remains a significant concern. In this work, we explore a frustratingly simple yet effective attack setting to test whether LLMs can be easily misled. Specifically, we evaluate LLMs on basic arithmetic questions (e.g., "What is 3 + 2?") presented as either multiple-choice or true-false judgment problems within PDF files, where hidden prompts are injected into the file. Our results reveal that LLMs are indeed vulnerable to such hidden prompt injection attacks, even in these trivial scenarios, highlighting serious robustness risks for LLM-as-a-judge applications.

摘要: 大型语言模型（LLM）最近在复杂推理和零次概括方面表现出了强大的涌现能力，在教育、同行评审和数据质量评估方面展示了LLM作为法官的应用前所未有的潜力。然而，它们在即时注入攻击（恶意指令被嵌入到内容中以操纵输出）下的稳健性仍然是一个值得关注的问题。在这项工作中，我们探索了一种令人沮丧的简单但有效的攻击设置，以测试LLM是否容易被误导。具体来说，我们评估基本算术问题的LLM（例如，“什么是3 + 2？”）在PDF文件中呈现为多项选择或真假判断问题，其中隐藏的提示被注入到文件中。我们的结果表明，即使在这些微不足道的场景中，LLM确实很容易受到此类隐藏的即时注入攻击，这凸显了LLM作为法官应用程序的严重鲁棒性风险。



## **30. Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection**

Emoji攻击：增强针对LLM法官检测的越狱攻击 cs.CL

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2411.01077v5) [paper-pdf](http://arxiv.org/pdf/2411.01077v5)

**Authors**: Zhipeng Wei, Yuqi Liu, N. Benjamin Erichson

**Abstract**: Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.

摘要: 越狱技术欺骗大型语言模型（LLM）产生受限输出，构成潜在威胁。一种防御措施是使用另一位LLM作为法官来评估生成文本的危害性。然而，我们发现这些Judge LLM很容易受到标记分割偏见的影响，当分隔符改变标记化过程、将单词分割成更小的子标记时，就会出现这个问题。这会改变整个序列的嵌入，降低检测准确性，并允许有害内容被错误分类为安全内容。在本文中，我们介绍了Emoji Attack，这是一种新颖的策略，通过利用代币分割偏见来放大现有的越狱提示。我们的方法利用上下文学习，在LLM法官评估文本之前系统地将表情符号插入文本中，从而引发嵌入失真，从而显着降低检测到不安全内容的可能性。与传统的分隔符不同，表情符号还会引入语义歧义，使它们在这种攻击中特别有效。通过对最先进的Judge LLM的实验，我们证明Emoji Attack大幅降低了不安全的预测率，绕过了现有的保障措施。



## **31. Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework**

LLM可以处理WebShell检测吗？使用行为功能感知框架克服检测挑战 cs.CR

Published as a conference paper at COLM 2025

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2504.13811v2) [paper-pdf](http://arxiv.org/pdf/2504.13811v2)

**Authors**: Feijiang Han, Jiaming Zhang, Chuyi Deng, Jianheng Tang, Yunhuai Liu

**Abstract**: WebShell attacks, where malicious scripts are injected into web servers, pose a significant cybersecurity threat. Traditional ML and DL methods are often hampered by challenges such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models have emerged as powerful alternatives for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that, stemming from their distinct analytical strategies, larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all baseline models lag behind previous SOTA methods. With the application of BFAD, the performance of all LLMs improves significantly, yielding an average F1 score increase of 13.82%. Notably, larger models now outperform SOTA benchmarks, while smaller models such as Qwen-2.5-Coder-3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection and provides solutions to address the challenges in this task.

摘要: WebShell攻击（恶意脚本被注入网络服务器）构成了重大的网络安全威胁。传统的ML和DL方法经常受到需要大量训练数据、灾难性遗忘和较差的概括性等挑战的阻碍。最近，大型语言模型已成为代码相关任务的强大替代方案，但它们在WebShell检测中的潜力仍然没有得到充分的开发。在本文中，我们做出了两项贡献：（1）对七种LLM进行了全面评估，包括GPT-4、LLaMA 3.1 70 B和Qwen 2.5变体，使用26.59 K个PHP脚本的数据集针对传统的基于序列和图形的方法进行基准测试，以及（2）行为功能感知检测（BFAD）框架，旨在解决将LLM应用于该领域的特定挑战。我们的框架集成了三个组件：隔离恶意PHP函数调用的关键函数过滤器、捕获最具行为指示性的代码段的上下文感知代码提取策略，以及通过优先考虑最相关的演示来增强上下文学习的加权行为函数剖析基于区分性功能级配置文件。我们的结果表明，由于其不同的分析策略，较大的LLM可以实现近乎完美的精确度，但召回率较低，而较小的模型则表现出相反的权衡。然而，所有基线模型都落后于之前的SOTA方法。随着BFAD的应用，所有LLM的性能都显着提高，F1平均得分提高了13.82%。值得注意的是，大型型号的性能现在优于SOTA基准，而Qwen-2.5-Coder-3B等小型型号的性能与传统方法相比具有竞争力。这项工作是第一个探索LLM用于WebShell检测的可行性和局限性的工作，并提供了解决方案来应对这项任务中的挑战。



## **32. Mitigating Jailbreaks with Intent-Aware LLMs**

利用意图意识的法学硕士缓解越狱 cs.CR

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2508.12072v1) [paper-pdf](http://arxiv.org/pdf/2508.12072v1)

**Authors**: Wei Jie Yeo, Ranjan Satapathy, Erik Cambria

**Abstract**: Despite extensive safety-tuning, large language models (LLMs) remain vulnerable to jailbreak attacks via adversarially crafted instructions, reflecting a persistent trade-off between safety and task performance. In this work, we propose Intent-FT, a simple and lightweight fine-tuning approach that explicitly trains LLMs to infer the underlying intent of an instruction before responding. By fine-tuning on a targeted set of adversarial instructions, Intent-FT enables LLMs to generalize intent deduction to unseen attacks, thereby substantially improving their robustness. We comprehensively evaluate both parametric and non-parametric attacks across open-source and proprietary models, considering harmfulness from attacks, utility, over-refusal, and impact against white-box threats. Empirically, Intent-FT consistently mitigates all evaluated attack categories, with no single attack exceeding a 50\% success rate -- whereas existing defenses remain only partially effective. Importantly, our method preserves the model's general capabilities and reduces excessive refusals on benign instructions containing superficially harmful keywords. Furthermore, models trained with Intent-FT accurately identify hidden harmful intent in adversarial attacks, and these learned intentions can be effectively transferred to enhance vanilla model defenses.

摘要: 尽管进行了广泛的安全调整，大型语言模型（LLM）仍然容易受到通过敌对设计的指令的越狱攻击，这反映了安全性和任务性能之间的持续权衡。在这项工作中，我们提出了Intent-FT，这是一种简单且轻量级的微调方法，它在响应之前显式训练LLM推断指令的潜在意图。通过对目标对抗指令集进行微调，Intent-FT使LLM能够将意图演绎推广到不可见的攻击，从而大幅提高其稳健性。我们全面评估开源和专有模型中的参数和非参数攻击，考虑攻击的危害性、效用、过度拒绝以及对白盒威胁的影响。从经验上看，Intent-FT始终如一地减轻了所有评估的攻击类别，没有一次攻击的成功率超过50%，而现有的防御措施仅保持部分有效。重要的是，我们的方法保留了模型的一般功能，并减少了对包含表面有害关键词的良性指令的过度拒绝。此外，使用Intent-FT训练的模型可以准确识别对抗性攻击中隐藏的有害意图，并且可以有效地转移这些习得的意图以增强普通模型防御。



## **33. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

MCA-Bench：评估CAPTCHA针对基于VLM的攻击的稳健性的多模式基准 cs.CV

we update the paper, add more experiments, and update the teammates

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2506.05982v4) [paper-pdf](http://arxiv.org/pdf/2506.05982v4)

**Authors**: Zonglin Wu, Yule Xue, Yaoyao Feng, Xiaolong Wang, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.

摘要: 随着自动攻击技术的迅速发展，验证码仍然是针对恶意机器人的重要防御机制。然而，现有的CAPTCHA方案涵盖了多种形式--从静态扭曲文本和模糊图像到交互式点击、滑动谜题和基于逻辑的问题--但社区仍然缺乏统一的、大规模的、多模式基准来严格评估其安全稳健性。为了解决这一差距，我们引入了MCA-Bench，这是一个全面且可重复的基准测试套件，可将异类CAPTCHA类型集成到单个评估协议中。利用共享的视觉语言模型主干，我们为每个CAPTCHA类别微调专门的破解剂，实现一致的跨模式评估。大量实验表明，MCA-Bench有效地绘制了现代CAPTCHA设计在不同攻击环境下的脆弱性谱，并且至关重要地提供了挑战复杂性、交互深度和模型可解性如何相互关联的首次定量分析。基于这些发现，我们提出了三项可操作的设计原则，并确定了关键的开放挑战，为系统性CAPTCHA强化、公平的基准测试和更广泛的社区合作奠定了基础。数据集和代码可在线获取。



## **34. MCP-Guard: A Defense Framework for Model Context Protocol Integrity in Large Language Model Applications**

MCP-Guard：大型语言模型应用中模型上下文协议完整性的防御框架 cs.CR

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10991v1) [paper-pdf](http://arxiv.org/pdf/2508.10991v1)

**Authors**: Wenpeng Xing, Zhonghao Qi, Yupeng Qin, Yilin Li, Caini Chang, Jiahui Yu, Changting Lin, Zhenzhen Xie, Meng Han

**Abstract**: The integration of Large Language Models (LLMs) with external tools via protocols such as the Model Context Protocol (MCP) introduces critical security vulnerabilities, including prompt injection, data exfiltration, and other threats. To counter these challenges, we propose MCP-Guard, a robust, layered defense architecture designed for LLM--tool interactions. MCP-Guard employs a three-stage detection pipeline that balances efficiency with accuracy: it progresses from lightweight static scanning for overt threats and a deep neural detector for semantic attacks, to our fine-tuned E5-based model achieves (96.01) accuracy in identifying adversarial prompts. Finally, a lightweight LLM arbitrator synthesizes these signals to deliver the final decision while minimizing false positives. To facilitate rigorous training and evaluation, we also introduce MCP-AttackBench, a comprehensive benchmark of over 70,000 samples. Sourced from public datasets and augmented by GPT-4, MCP-AttackBench simulates diverse, real-world attack vectors in the MCP format, providing a foundation for future research into securing LLM-tool ecosystems.

摘要: 通过模型上下文协议（HCP）等协议将大型语言模型（LLM）与外部工具集成会引入严重的安全漏洞，包括提示注入、数据溢出和其他威胁。为了应对这些挑战，我们提出了MCP-Guard，这是一种专为LLM工具交互而设计的稳健、分层的防御架构。MCP-Guard采用三阶段检测管道，平衡效率与准确性：它从针对明显威胁的轻量级静态扫描和针对语义攻击的深度神经检测器，发展到我们微调的基于E5的模型，在识别对抗性提示方面实现了（96.01）的准确性。最后，轻量级LLM仲裁器合成这些信号以做出最终决策，同时最大限度地减少误报。为了促进严格的培训和评估，我们还引入了MCP-AttackBench，这是一个包含超过70，000个样本的综合基准。MCP-AttackBench源自公共数据集，并通过GPT-4进行增强，以HCP格式模拟不同的现实世界攻击载体，为未来研究保护LLM工具生态系统提供基础。



## **35. Failures to Surface Harmful Contents in Video Large Language Models**

未能在视频大语言模型中暴露有害内容 cs.MM

11 pages, 8 figures

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10974v1) [paper-pdf](http://arxiv.org/pdf/2508.10974v1)

**Authors**: Yuxin Cao, Wei Song, Derui Wang, Jingling Xue, Jin Song Dong

**Abstract**: Video Large Language Models (VideoLLMs) are increasingly deployed on numerous critical applications, where users rely on auto-generated summaries while casually skimming the video stream. We show that this interaction hides a critical safety gap: if harmful content is embedded in a video, either as full-frame inserts or as small corner patches, state-of-the-art VideoLLMs rarely mention the harmful content in the output, despite its clear visibility to human viewers. A root-cause analysis reveals three compounding design flaws: (1) insufficient temporal coverage resulting from the sparse, uniformly spaced frame sampling used by most leading VideoLLMs, (2) spatial information loss introduced by aggressive token downsampling within sampled frames, and (3) encoder-decoder disconnection, whereby visual cues are only weakly utilized during text generation. Leveraging these insights, we craft three zero-query black-box attacks, aligning with these flaws in the processing pipeline. Our large-scale evaluation across five leading VideoLLMs shows that the harmfulness omission rate exceeds 90% in most cases. Even when harmful content is clearly present in all frames, these models consistently fail to identify it. These results underscore a fundamental vulnerability in current VideoLLMs' designs and highlight the urgent need for sampling strategies, token compression, and decoding mechanisms that guarantee semantic coverage rather than speed alone.

摘要: 视频大型语言模型（VideoLLM）越来越多地部署在许多关键应用程序上，其中用户依赖自动生成的摘要，同时随意浏览视频流。我们表明，这种交互隐藏着一个关键的安全差距：如果有害内容嵌入视频中，无论是作为全帧插入还是作为小角补丁，那么最先进的VideoLLM很少在输出中提及有害内容，尽管它对人类观众来说是清晰可见的。根本原因分析揭示了三个复合设计缺陷：（1）大多数领先的VideoLLM使用的稀疏、均匀间隔的帧采样导致的时间覆盖不足，（2）采样帧内的激进令牌下采样引入的空间信息丢失，以及（3）编码器-解码器断开连接，从而视觉线索在文本生成过程中仅被微弱地利用。利用这些见解，我们设计了三种零查询黑匣子攻击，以与处理管道中的这些缺陷保持一致。我们对五家领先的VideoLLM进行的大规模评估显示，在大多数情况下，危害性遗漏率超过90%。即使有害内容明显存在于所有帧中，这些模型仍然无法识别它。这些结果强调了当前VideoLLM设计中的一个根本漏洞，并强调了对保证语义覆盖而不仅仅是速度的采样策略、令牌压缩和解码机制的迫切需要。



## **36. An Explainable Transformer-based Model for Phishing Email Detection: A Large Language Model Approach**

用于网络钓鱼电子邮件检测的可解释的基于转换器的模型：大语言模型方法 cs.LG

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2402.13871v2) [paper-pdf](http://arxiv.org/pdf/2402.13871v2)

**Authors**: Mohammad Amaz Uddin, Md Mahiuddin, Iqbal H. Sarker

**Abstract**: Phishing email is a serious cyber threat that tries to deceive users by sending false emails with the intention of stealing confidential information or causing financial harm. Attackers, often posing as trustworthy entities, exploit technological advancements and sophistication to make detection and prevention of phishing more challenging. Despite extensive academic research, phishing detection remains an ongoing and formidable challenge in the cybersecurity landscape. Large Language Models (LLMs) and Masked Language Models (MLMs) possess immense potential to offer innovative solutions to address long-standing challenges. In this research paper, we present an optimized, fine-tuned transformer-based DistilBERT model designed for the detection of phishing emails. In the detection process, we work with a phishing email dataset and utilize the preprocessing techniques to clean and solve the imbalance class issues. Through our experiments, we found that our model effectively achieves high accuracy, demonstrating its capability to perform well. Finally, we demonstrate our fine-tuned model using Explainable-AI (XAI) techniques such as Local Interpretable Model-Agnostic Explanations (LIME) and Transformer Interpret to explain how our model makes predictions in the context of text classification for phishing emails.

摘要: 网络钓鱼电子邮件是一种严重的网络威胁，试图通过发送虚假电子邮件来欺骗用户，意图窃取机密信息或造成经济损失。攻击者通常冒充值得信赖的实体，利用技术进步和复杂性使网络钓鱼的检测和预防更具挑战性。尽管进行了广泛的学术研究，但网络钓鱼检测仍然是网络安全领域持续且艰巨的挑战。大型语言模型（LLM）和掩蔽语言模型（MLM）具有提供创新解决方案来应对长期挑战的巨大潜力。在这篇研究论文中，我们提出了一个优化、微调的基于变压器的DistilBERT模型，旨在检测网络钓鱼电子邮件。在检测过程中，我们使用网络钓鱼电子邮件数据集，并利用预处理技术来清理和解决不平衡类别问题。通过实验，我们发现我们的模型有效地实现了高准确性，证明了其性能良好的能力。最后，我们使用可解释人工智能（XAI）技术（例如本地可解释模型不可知解释（LIME）和Transformer Interpret）演示了我们的微调模型，以解释我们的模型如何在网络钓鱼电子邮件的文本分类背景下做出预测。



## **37. Enhancing GraphQL Security by Detecting Malicious Queries Using Large Language Models, Sentence Transformers, and Convolutional Neural Networks**

通过使用大型语言模型、句子转换器和卷积神经网络检测恶意收件箱来增强GraphQL安全性 cs.CR

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.11711v1) [paper-pdf](http://arxiv.org/pdf/2508.11711v1)

**Authors**: Irash Perera, Hiranya Abeyrathne, Sanjeewa Malalgoda, Arshardh Ifthikar

**Abstract**: GraphQL's flexibility, while beneficial for efficient data fetching, introduces unique security vulnerabilities that traditional API security mechanisms often fail to address. Malicious GraphQL queries can exploit the language's dynamic nature, leading to denial-of-service attacks, data exfiltration through injection, and other exploits. Existing solutions, such as static analysis, rate limiting, and general-purpose Web Application Firewalls, offer limited protection against sophisticated, context-aware attacks. This paper presents a novel, AI-driven approach for real-time detection of malicious GraphQL queries. Our method combines static analysis with machine learning techniques, including Large Language Models (LLMs) for dynamic schema-based configuration, Sentence Transformers (SBERT and Doc2Vec) for contextual embedding of query payloads, and Convolutional Neural Networks (CNNs), Random Forests, and Multilayer Perceptrons for classification. We detail the system architecture, implementation strategies optimized for production environments (including ONNX Runtime optimization and parallel processing), and evaluate the performance of our detection models and the overall system under load. Results demonstrate high accuracy in detecting various threats, including SQL injection, OS command injection, and XSS exploits, alongside effective mitigation of DoS and SSRF attempts. This research contributes a robust and adaptable solution for enhancing GraphQL API security.

摘要: GraphQL的灵活性虽然有利于高效的数据获取，但也引入了传统API安全机制通常无法解决的独特安全漏洞。恶意的GraphQL查询可以利用该语言的动态性质，导致拒绝服务攻击、通过注入的数据泄露和其他利用。现有的解决方案（例如静态分析、速率限制和通用Web应用程序防火墙）只能针对复杂的上下文感知攻击提供有限的保护。本文提出了一种新颖的人工智能驱动方法，用于实时检测恶意GraphQL查询。我们的方法将静态分析与机器学习技术相结合，包括用于基于动态模式的配置的大型语言模型（LLM）、用于查询有效负载的上下文嵌入的句子转换器（SBERT和Doc 2Vec），以及用于分类的卷积神经网络（CNN）、随机森林和多层感知器。我们详细介绍了针对生产环境优化的系统架构、实施策略（包括ONNX收件箱优化和并行处理），并评估我们的检测模型和整个系统负载下的性能。结果表明，在检测各种威胁（包括SQL注入、OS命令注入和XSS漏洞利用）方面具有高准确性，并且有效缓解了DPS和SSRF尝试。这项研究为增强GraphQL API安全性提供了一个强大且适应性强的解决方案。



## **38. Layer-Wise Perturbations via Sparse Autoencoders for Adversarial Text Generation**

通过稀疏自动编码器进行分层扰动以生成对抗性文本 cs.CL

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10404v1) [paper-pdf](http://arxiv.org/pdf/2508.10404v1)

**Authors**: Huizhen Shu, Xuying Li, Qirui Wang, Yuji Kosuga, Mengqiu Tian, Zhuo Li

**Abstract**: With the rapid proliferation of Natural Language Processing (NLP), especially Large Language Models (LLMs), generating adversarial examples to jailbreak LLMs remains a key challenge for understanding model vulnerabilities and improving robustness. In this context, we propose a new black-box attack method that leverages the interpretability of large models. We introduce the Sparse Feature Perturbation Framework (SFPF), a novel approach for adversarial text generation that utilizes sparse autoencoders to identify and manipulate critical features in text. After using the SAE model to reconstruct hidden layer representations, we perform feature clustering on the successfully attacked texts to identify features with higher activations. These highly activated features are then perturbed to generate new adversarial texts. This selective perturbation preserves the malicious intent while amplifying safety signals, thereby increasing their potential to evade existing defenses. Our method enables a new red-teaming strategy that balances adversarial effectiveness with safety alignment. Experimental results demonstrate that adversarial texts generated by SFPF can bypass state-of-the-art defense mechanisms, revealing persistent vulnerabilities in current NLP systems.However, the method's effectiveness varies across prompts and layers, and its generalizability to other architectures and larger models remains to be validated.

摘要: 随着自然语言处理（NLP），尤其是大型语言模型（LLM）的迅速普及，生成对抗性示例以越狱LLM仍然是理解模型漏洞和提高稳健性的关键挑战。在此背景下，我们提出了一种新的黑匣子攻击方法，该方法利用了大型模型的可解释性。我们引入了稀疏特征扰动框架（SPF），这是一种对抗性文本生成的新颖方法，利用稀疏自动编码器来识别和操纵文本中的关键特征。在使用SAGE模型重建隐藏层表示后，我们对成功攻击的文本执行特征集群，以识别激活程度较高的特征。然后，这些高度激活的特征被扰动以生成新的对抗文本。这种选择性干扰在放大安全信号的同时保留了恶意意图，从而增加了它们逃避现有防御的可能性。我们的方法实现了一种新的红色团队策略，该策略平衡了对抗有效性与安全一致。实验结果表明，SFPF生成的对抗性文本可以绕过最先进的防御机制，揭示当前NLP系统中的持久漏洞。然而，该方法的有效性因提示和层而异，其对其他架构和更大模型的推广性仍有待验证。



## **39. Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts**

越狱商业黑匣子法学硕士，带有明显有害的承诺 cs.CL

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10390v1) [paper-pdf](http://arxiv.org/pdf/2508.10390v1)

**Authors**: Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Liming Fang, Zhe Liu

**Abstract**: Evaluating jailbreak attacks is challenging when prompts are not overtly harmful or fail to induce harmful outputs. Unfortunately, many existing red-teaming datasets contain such unsuitable prompts. To evaluate attacks accurately, these datasets need to be assessed and cleaned for maliciousness. However, existing malicious content detection methods rely on either manual annotation, which is labor-intensive, or large language models (LLMs), which have inconsistent accuracy in harmful types. To balance accuracy and efficiency, we propose a hybrid evaluation framework named MDH (Malicious content Detection based on LLMs with Human assistance) that combines LLM-based annotation with minimal human oversight, and apply it to dataset cleaning and detection of jailbroken responses. Furthermore, we find that well-crafted developer messages can significantly boost jailbreak success, leading us to propose two new strategies: D-Attack, which leverages context simulation, and DH-CoT, which incorporates hijacked chains of thought. The Codes, datasets, judgements, and detection results will be released in github repository: https://github.com/AlienZhang1996/DH-CoT.

摘要: 当提示没有明显有害或未能引发有害输出时，评估越狱攻击具有挑战性。不幸的是，许多现有的红色团队数据集包含此类不合适的提示。为了准确评估攻击，需要评估和清理这些数据集的恶意性。然而，现有的恶意内容检测方法要么依赖于劳动密集型的手动注释，要么依赖于大型语言模型（LLM），后者在有害类型中的准确性不一致。为了平衡准确性和效率，我们提出了一个名为MDH（基于LLM与人工辅助的恶意内容检测）的混合评估框架，该框架将基于LLM的注释与最少的人为监督相结合，并将其应用于数据集清理和越狱响应的检测。此外，我们发现精心制作的开发人员消息可以显着提高越狱成功率，这使得我们提出了两种新的策略：D-Attack，它利用上下文模拟，以及DH-CoT，它结合了劫持的思想链。代码，数据集，判断和检测结果将在github存储库中发布：https://github.com/AlienZhang1996/DH-CoT。



## **40. A Vision-Language Pre-training Model-Guided Approach for Mitigating Backdoor Attacks in Federated Learning**

一种视觉语言预训练模型引导的方法用于缓解联邦学习中的后门攻击 cs.LG

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10315v1) [paper-pdf](http://arxiv.org/pdf/2508.10315v1)

**Authors**: Keke Gai, Dongjue Wang, Jing Yu, Liehuang Zhu, Qi Wu

**Abstract**: Existing backdoor defense methods in Federated Learning (FL) rely on the assumption of homogeneous client data distributions or the availability of a clean serve dataset, which limits the practicality and effectiveness. Defending against backdoor attacks under heterogeneous client data distributions while preserving model performance remains a significant challenge. In this paper, we propose a FL backdoor defense framework named CLIP-Fed, which leverages the zero-shot learning capabilities of vision-language pre-training models. By integrating both pre-aggregation and post-aggregation defense strategies, CLIP-Fed overcomes the limitations of Non-IID imposed on defense effectiveness. To address privacy concerns and enhance the coverage of the dataset against diverse triggers, we construct and augment the server dataset using the multimodal large language model and frequency analysis without any client samples. To address class prototype deviations caused by backdoor samples and eliminate the correlation between trigger patterns and target labels, CLIP-Fed aligns the knowledge of the global model and CLIP on the augmented dataset using prototype contrastive loss and Kullback-Leibler divergence. Extensive experiments on representative datasets validate the effectiveness of CLIP-Fed. Compared to state-of-the-art methods, CLIP-Fed achieves an average reduction in ASR, i.e., 2.03\% on CIFAR-10 and 1.35\% on CIFAR-10-LT, while improving average MA by 7.92\% and 0.48\%, respectively.

摘要: 联邦学习（FL）中现有的后门防御方法依赖于同质客户端数据分布或干净服务数据集的可用性的假设，这限制了实用性和有效性。防御异类客户端数据分布下的后门攻击，同时保持模型性能仍然是一个重大挑战。在本文中，我们提出了一个名为CLIP-Fed的FL后门防御框架，该框架利用了视觉语言预训练模型的零射击学习能力。通过整合前聚合和后聚合防御策略，CLIP-Fed克服了非IID对防御有效性的限制。为了解决隐私问题并增强数据集针对不同触发因素的覆盖范围，我们使用多模式大型语言模型和频率分析来构建和增强服务器数据集，而无需任何客户端样本。为了解决后门样本引起的类原型偏差并消除触发模式和目标标签之间的相关性，CLIP-Fed使用原型对比损失和Kullback-Leibler分歧将全局模型和CLIP的知识整合在增强数据集中。对代表性数据集的大量实验验证了CLIP-Fed的有效性。与最先进的方法相比，CLIP-Fed实现了ASB的平均降低，即CIFAR-10和CIFAR-10-LT的平均MA分别提高了2.03%和1.35%，平均MA分别提高了7.92%和0.48%。



## **41. Extending the OWASP Multi-Agentic System Threat Modeling Guide: Insights from Multi-Agent Security Research**

扩展OWSP多统计系统威胁建模指南：来自多代理安全研究的见解 cs.MA

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.09815v1) [paper-pdf](http://arxiv.org/pdf/2508.09815v1)

**Authors**: Klaudia Krawiecka, Christian Schroeder de Witt

**Abstract**: We propose an extension to the OWASP Multi-Agentic System (MAS) Threat Modeling Guide, translating recent anticipatory research in multi-agent security (MASEC) into practical guidance for addressing challenges unique to large language model (LLM)-driven multi-agent architectures. Although OWASP's existing taxonomy covers many attack vectors, our analysis identifies gaps in modeling failures, including, but not limited to: reasoning collapse across planner-executor chains, metric overfitting, unsafe delegation escalation, emergent covert coordination, and heterogeneous multi-agent exploits. We introduce additional threat classes and scenarios grounded in practical MAS deployments, highlighting risks from benign goal drift, cross-agent hallucination propagation, affective prompt framing, and multi-agent backdoors. We also outline evaluation strategies, including robustness testing, coordination assessment, safety enforcement, and emergent behavior monitoring, to ensure complete coverage. This work complements the framework of OWASP by expanding its applicability to increasingly complex, autonomous, and adaptive multi-agent systems, with the goal of improving security posture and resilience in real world deployments.

摘要: 我们提出了一个扩展OWASP多智能体系统（MAS）威胁建模指南，翻译最近的预期研究多智能体安全（MASEC）到实用的指导，以解决独特的大语言模型（LLM）驱动的多智能体架构的挑战。虽然OWASP现有的分类法涵盖了许多攻击向量，但我们的分析确定了建模失败的差距，包括但不限于：规划者-执行者链的推理崩溃，度量过拟合，不安全的委托升级，紧急隐蔽协调和异构多代理漏洞。我们介绍了额外的威胁类和场景接地实际MAS部署，突出良性的目标漂移，跨代理幻觉传播，情感提示框架和多代理后门的风险。我们还概述了评估策略，包括鲁棒性测试，协调评估，安全执法和紧急行为监测，以确保完全覆盖。这项工作通过将OWISP的适用性扩展到日益复杂、自治和自适应的多代理系统，补充了OWISP的框架，目标是改善现实世界部署中的安全姿态和弹性。



## **42. MetaCipher: A Time-Persistent and Universal Multi-Agent Framework for Cipher-Based Jailbreak Attacks for LLMs**

MetaCipher：针对LLM的基于密码的越狱攻击的持续时间和通用多代理框架 cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2506.22557v2) [paper-pdf](http://arxiv.org/pdf/2506.22557v2)

**Authors**: Boyuan Chen, Minghao Shao, Abdul Basit, Siddharth Garg, Muhammad Shafique

**Abstract**: As large language models (LLMs) grow more capable, they face growing vulnerability to sophisticated jailbreak attacks. While developers invest heavily in alignment finetuning and safety guardrails, researchers continue publishing novel attacks, driving progress through adversarial iteration. This dynamic mirrors a strategic game of continual evolution. However, two major challenges hinder jailbreak development: the high cost of querying top-tier LLMs and the short lifespan of effective attacks due to frequent safety updates. These factors limit cost-efficiency and practical impact of research in jailbreak attacks. To address this, we propose MetaCipher, a low-cost, multi-agent jailbreak framework that generalizes across LLMs with varying safety measures. Using reinforcement learning, MetaCipher is modular and adaptive, supporting extensibility to future strategies. Within as few as 10 queries, MetaCipher achieves state-of-the-art attack success rates on recent malicious prompt benchmarks, outperforming prior jailbreak methods. We conduct a large-scale empirical evaluation across diverse victim models and benchmarks, demonstrating its robustness and adaptability. Warning: This paper contains model outputs that may be offensive or harmful, shown solely to demonstrate jailbreak efficacy.

摘要: 随着大型语言模型（LLM）的能力变得越来越强，它们面临着越来越容易受到复杂越狱攻击的脆弱性。虽然开发人员在对齐微调和安全护栏上投入巨资，但研究人员继续发布新颖的攻击，通过对抗迭代推动进展。这种动态反映了一场持续进化的战略游戏。然而，有两个主要挑战阻碍了越狱的发展：查询顶级LLM的高成本以及由于频繁的安全更新而导致有效攻击的寿命短。这些因素限制了越狱攻击研究的成本效率和实际影响。为了解决这个问题，我们提出了MetaCipher，这是一种低成本、多代理越狱框架，可在具有不同安全措施的LLM中进行推广。使用强化学习，MetaCipher具有模块化和自适应性，支持未来策略的可扩展性。在短短10个查询内，MetaCipher就在最近的恶意提示基准上实现了最先进的攻击成功率，优于之前的越狱方法。我们对不同的受害者模型和基准进行了大规模的实证评估，展示了其稳健性和适应性。警告：本文包含可能令人反感或有害的模型输出，仅用于证明越狱功效。



## **43. Guardians and Offenders: A Survey on Harmful Content Generation and Safety Mitigation of LLM**

监护人和罪犯：关于LLM有害内容生成和安全缓解的调查 cs.CL

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.05775v2) [paper-pdf](http://arxiv.org/pdf/2508.05775v2)

**Authors**: Chi Zhang, Changjia Zhu, Junjie Xiong, Xiaoran Xu, Lingyao Li, Yao Liu, Zhuo Lu

**Abstract**: Large Language Models (LLMs) have revolutionized content creation across digital platforms, offering unprecedented capabilities in natural language generation and understanding. These models enable beneficial applications such as content generation, question and answering (Q&A), programming, and code reasoning. Meanwhile, they also pose serious risks by inadvertently or intentionally producing toxic, offensive, or biased content. This dual role of LLMs, both as powerful tools for solving real-world problems and as potential sources of harmful language, presents a pressing sociotechnical challenge. In this survey, we systematically review recent studies spanning unintentional toxicity, adversarial jailbreaking attacks, and content moderation techniques. We propose a unified taxonomy of LLM-related harms and defenses, analyze emerging multimodal and LLM-assisted jailbreak strategies, and assess mitigation efforts, including reinforcement learning with human feedback (RLHF), prompt engineering, and safety alignment. Our synthesis highlights the evolving landscape of LLM safety, identifies limitations in current evaluation methodologies, and outlines future research directions to guide the development of robust and ethically aligned language technologies.

摘要: 大型语言模型（LLM）彻底改变了数字平台上的内容创建，在自然语言生成和理解方面提供了前所未有的能力。这些模型支持有益的应用程序，如内容生成、问答（Q&A）、编程和代码推理。与此同时，它们也会因无意或故意产生有毒、攻击性或有偏见的内容而构成严重风险。LLM的这种双重角色，既作为解决现实世界问题的强大工具，又作为有害语言的潜在来源，提出了一个紧迫的社会技术挑战。在这项调查中，我们系统地回顾了最近的研究，涵盖无意毒性、对抗性越狱攻击和内容审核技术。我们提出了LLM相关伤害和防御的统一分类，分析新兴的多模式和LLM辅助越狱策略，并评估缓解工作，包括人类反馈强化学习（RL HF）、即时工程和安全调整。我们的综合强调了LLM安全性不断变化的格局，确定了当前评估方法的局限性，并概述了未来的研究方向，以指导稳健且符合道德规范的语言技术的开发。



## **44. NeuronTune: Fine-Grained Neuron Modulation for Balanced Safety-Utility Alignment in LLMs**

NeuronButton：细粒度神经元调制，实现LLM中平衡的安全-效用对齐 cs.LG

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.09473v1) [paper-pdf](http://arxiv.org/pdf/2508.09473v1)

**Authors**: Birong Pan, Mayi Xu, Qiankun Pi, Jianhao Chen, Yuanyuan Zhu, Ming Zhong, Tieyun Qian

**Abstract**: Ensuring robust safety alignment while preserving utility is critical for the reliable deployment of Large Language Models (LLMs). However, current techniques fundamentally suffer from intertwined deficiencies: insufficient robustness against malicious attacks, frequent refusal of benign queries, degradation in generated text quality and general task performance--the former two reflecting deficits in robust safety and the latter constituting utility impairment. We trace these limitations to the coarse-grained layer-wise interventions in existing methods. To resolve this, we propose NeuronTune, a fine-grained framework that dynamically modulates sparse neurons to achieve simultaneous safety-utility optimization. Our approach first identifies safety-critical and utility-preserving neurons across all layers via attribution, then employs meta-learning to adaptively amplify safety-neuron activations and suppress utility-neuron activations. Crucially, NeuronTune enables tunable adjustment of intervention scope via neuron-count thresholds, supporting flexible adaptation to security-critical or utility-priority scenarios. Extensive experimental results demonstrate that our method significantly outperforms existing state-of-the-art technologies, achieving superior model safety while maintaining excellent utility.

摘要: 确保稳健的安全一致同时保持实用性对于大型语言模型（LLM）的可靠部署至关重要。然而，当前的技术从根本上来说存在着相互交织的缺陷：针对恶意攻击的鲁棒性不足、频繁拒绝良性查询、生成的文本质量和一般任务性能下降--前两者反映了鲁棒安全性的缺陷，后者构成了效用损害。我们将这些限制追溯到现有方法中的粗粒度分层干预。为了解决这个问题，我们提出了NeuronButton，这是一个细粒度框架，可以动态调节稀疏神经元以实现同时的安全-效用优化。我们的方法首先通过归因识别所有层中的安全关键和效用保留神经元，然后采用元学习来自适应地放大安全神经元激活并抑制效用神经元激活。至关重要的是，NeuronButton可以通过神经元计数阈值对干预范围进行可调调整，支持灵活适应安全关键或公用事业优先场景。大量的实验结果表明，我们的方法显着优于现有的最先进技术，在保持出色的实用性的同时实现了卓越的模型安全性。



## **45. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟抓住漏洞：揭开LLM服务系统中的计时侧通道 cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects  the up-to-date experimental results

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2409.20002v4) [paper-pdf](http://arxiv.org/pdf/2409.20002v4)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型（LLM）的广泛部署引发了对其推理性能优化的强烈要求。当今服务于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，同时在很大程度上忽视了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道由共享缓存和图形处理器内存分配产生，可以利用这些通道来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了传统计算系统中观察到的安全挑战，凸显了解决LLM服务基础设施中潜在信息泄露的迫切需要。在本文中，我们报告了旨在利用LLM部署中固有的此类时间侧通道的新颖攻击策略，特别针对广泛用于增强LLM推理性能的Key-Value（KV）缓存和语义缓存。我们的方法利用时间测量和分类模型来检测缓存命中，使对手能够高准确地推断私人提示。我们还提出了一种逐令牌搜索算法来有效地恢复缓存中的共享提示前置，展示了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑匣子测试的实验研究表明，此类隐私风险是完全现实的，并会产生重大后果。我们的研究结果强调需要强有力的缓解措施来保护LLM系统免受此类新出现的威胁。



## **46. Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference**

缓存中的影子：在LLM推理中揭示和减轻KV缓存的隐私风险 cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.09442v1) [paper-pdf](http://arxiv.org/pdf/2508.09442v1)

**Authors**: Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou, Yuke Hu, Chenxu Zhao, Zhihao Liu, Zhan Qin

**Abstract**: The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.

摘要: Key-Value（KV）缓存存储中间注意力计算（Key和Value对）以避免冗余计算，是加速大型语言模型（LLM）推理的基本机制。然而，这种效率优化引入了重大但未充分探索的隐私风险。本文首次对这些漏洞进行了全面分析，证明攻击者可以直接从KV缓存重建敏感用户输入。我们设计并实现了三种不同的攻击载体：直接倒置攻击、更广泛适用且更强大的碰撞攻击以及基于语义的注入攻击。这些方法证明了KV缓存隐私泄漏问题的实用性和严重性。为了减轻这种情况，我们提出了KV-斗篷，一种新颖的，轻量级的，高效的防御机制。KV-Cloak使用基于可逆矩阵的混淆方案，结合运算符融合来保护KV缓存。我们广泛的实验表明，KV-斗篷有效地挫败了所有提出的攻击，降低重建质量的随机噪声。至关重要的是，它实现了这种强大的安全性，模型准确性几乎没有下降，性能负担最小，为值得信赖的LLM部署提供了实用的解决方案。



## **47. Attacks and Defenses Against LLM Fingerprinting**

针对LLM指纹的攻击和防御 cs.CR

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.09021v1) [paper-pdf](http://arxiv.org/pdf/2508.09021v1)

**Authors**: Kevin Kurian, Ethan Holland, Sean Oesch

**Abstract**: As large language models are increasingly deployed in sensitive environments, fingerprinting attacks pose significant privacy and security risks. We present a study of LLM fingerprinting from both offensive and defensive perspectives. Our attack methodology uses reinforcement learning to automatically optimize query selection, achieving better fingerprinting accuracy with only 3 queries compared to randomly selecting 3 queries from the same pool. Our defensive approach employs semantic-preserving output filtering through a secondary LLM to obfuscate model identity while maintaining semantic integrity. The defensive method reduces fingerprinting accuracy across tested models while preserving output quality. These contributions show the potential to improve fingerprinting tools capabilities while providing practical mitigation strategies against fingerprinting attacks.

摘要: 随着大型语言模型越来越多地部署在敏感环境中，指纹攻击构成了巨大的隐私和安全风险。我们从进攻和防守的角度对LLM指纹进行了研究。我们的攻击方法使用强化学习来自动优化查询选择，与从同一池中随机选择3个查询相比，仅使用3个查询就可以实现更好的指纹识别准确性。我们的防御方法通过二级LLM采用保持语义的输出过滤来混淆模型身份，同时保持语义完整性。防御性方法降低了测试模型之间的指纹识别准确性，同时保留了输出质量。这些贡献表明了提高指纹识别工具功能的潜力，同时提供针对指纹识别攻击的实用缓解策略。



## **48. GUARD:Dual-Agent based Backdoor Defense on Chain-of-Thought in Neural Code Generation**

GUARD：基于双代理的神经代码生成思想链后门防御 cs.SE

Accepted by SEKE 2025

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2505.21425v3) [paper-pdf](http://arxiv.org/pdf/2505.21425v3)

**Authors**: Naizhu Jin, Zhong Li, Tian Zhang, Qingkai Zeng

**Abstract**: With the widespread application of large language models in code generation, recent studies demonstrate that employing additional Chain-of-Thought generation models can significantly enhance code generation performance by providing explicit reasoning steps. However, as external components, CoT models are particularly vulnerable to backdoor attacks, which existing defense mechanisms often fail to detect effectively. To address this challenge, we propose GUARD, a novel dual-agent defense framework specifically designed to counter CoT backdoor attacks in neural code generation. GUARD integrates two core components: GUARD-Judge, which identifies suspicious CoT steps and potential triggers through comprehensive analysis, and GUARD-Repair, which employs a retrieval-augmented generation approach to regenerate secure CoT steps for identified anomalies. Experimental results show that GUARD effectively mitigates attacks while maintaining generation quality, advancing secure code generation systems.

摘要: 随着大型语言模型在代码生成中的广泛应用，最近的研究表明，采用额外的思想链生成模型可以通过提供显式推理步骤来显着提高代码生成性能。然而，作为外部组件，CoT模型特别容易受到后门攻击，而现有的防御机制往往无法有效检测到后门攻击。为了应对这一挑战，我们提出了GUARD，这是一种新型双代理防御框架，专门设计用于对抗神经代码生成中的CoT后门攻击。GUARD集成了两个核心组件：GUARD-Judge，通过全面分析识别可疑的CoT步骤和潜在触发因素，以及GUARD-Repair，采用检索增强生成方法来为识别的异常重新生成安全CoT步骤。实验结果表明，GUARD有效地缓解了攻击，同时保持生成质量，推进了安全代码生成系统。



## **49. Whispers in the Machine: Confidentiality in Agentic Systems**

机器中的耳语：智能系统中的机密 cs.CR

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2402.06922v4) [paper-pdf](http://arxiv.org/pdf/2402.06922v4)

**Authors**: Jonathan Evertz, Merlin Chlosta, Lea Schönherr, Thorsten Eisenhofer

**Abstract**: The interaction between users and applications is increasingly shifted toward natural language by deploying Large Language Models (LLMs) as the core interface. The capabilities of these so-called agents become more capable the more tools and services they serve as an interface for, ultimately leading to agentic systems. Agentic systems use LLM-based agents as interfaces for most user interactions and various integrations with external tools and services. While these interfaces can significantly enhance the capabilities of the agentic system, they also introduce a new attack surface. Manipulated integrations, for example, can exploit the internal LLM and compromise sensitive data accessed through other interfaces. While previous work primarily focused on attacks targeting a model's alignment or the leakage of training data, the security of data that is only available during inference has escaped scrutiny so far. In this work, we demonstrate how the integration of LLMs into systems with external tool integration poses a risk similar to established prompt-based attacks, able to compromise the confidentiality of the entire system. Introducing a systematic approach to evaluate these confidentiality risks, we identify two specific attack scenarios unique to these agentic systems and formalize these into a tool-robustness framework designed to measure a model's ability to protect sensitive information. Our analysis reveals significant vulnerabilities across all tested models, highlighting an increased risk when models are combined with external tools.

摘要: 通过将大型语言模型（LLM）部署为核心接口，用户和应用程序之间的交互越来越多地转向自然语言。这些所谓的代理人的能力变得更有能力的工具和服务，他们作为一个接口，最终导致代理系统。智能系统使用基于LLM的代理作为大多数用户交互的接口以及与外部工具和服务的各种集成。虽然这些接口可以显著增强代理系统的能力，但它们也引入了新的攻击面。例如，经过操纵的集成可能会利用内部LLM并损害通过其他接口访问的敏感数据。虽然之前的工作主要集中在针对模型对齐或训练数据泄露的攻击上，但迄今为止仅在推理期间可用的数据安全性逃脱了审查。在这项工作中，我们展示了将LLM集成到具有外部工具集成的系统中如何构成类似于已建立的基于预算的攻击的风险，从而能够损害整个系统的机密性。我们引入了一种系统性方法来评估这些保密风险，识别了这些代理系统独有的两种特定攻击场景，并将其形式化为工具稳健性框架，旨在衡量模型保护敏感信息的能力。我们的分析揭示了所有测试模型中存在的重大漏洞，凸显了模型与外部工具结合时的风险增加。



## **50. A Few Words Can Distort Graphs: Knowledge Poisoning Attacks on Graph-based Retrieval-Augmented Generation of Large Language Models**

一些词可以扭曲图形：对基于图形的检索增强生成大型语言模型的知识中毒攻击 cs.CL

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.04276v2) [paper-pdf](http://arxiv.org/pdf/2508.04276v2)

**Authors**: Jiayi Wen, Tianxin Chen, Zhirun Zheng, Cheng Huang

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) has recently emerged as a promising paradigm for enhancing large language models (LLMs) by converting raw text into structured knowledge graphs, improving both accuracy and explainability. However, GraphRAG relies on LLMs to extract knowledge from raw text during graph construction, and this process can be maliciously manipulated to implant misleading information. Targeting this attack surface, we propose two knowledge poisoning attacks (KPAs) and demonstrate that modifying only a few words in the source text can significantly change the constructed graph, poison the GraphRAG, and severely mislead downstream reasoning. The first attack, named Targeted KPA (TKPA), utilizes graph-theoretic analysis to locate vulnerable nodes in the generated graphs and rewrites the corresponding narratives with LLMs, achieving precise control over specific question-answering (QA) outcomes with a success rate of 93.1\%, while keeping the poisoned text fluent and natural. The second attack, named Universal KPA (UKPA), exploits linguistic cues such as pronouns and dependency relations to disrupt the structural integrity of the generated graph by altering globally influential words. With fewer than 0.05\% of full text modified, the QA accuracy collapses from 95\% to 50\%. Furthermore, experiments show that state-of-the-art defense methods fail to detect these attacks, highlighting that securing GraphRAG pipelines against knowledge poisoning remains largely unexplored.

摘要: 基于图的检索增强生成（GraphRAG）最近成为一种有前途的范式，可以通过将原始文本转换为结构化知识图来增强大型语言模型（LLM），提高准确性和可解释性。然而，GraphRAG在图形构建过程中依赖LLM从原始文本中提取知识，而这个过程可能会被恶意操纵以植入误导性信息。针对这一攻击面，我们提出了两种知识中毒攻击（KPA），并证明仅修改源文本中的几个单词就可以显着改变所构建的图、毒害GraphRAG并严重误导下游推理。第一次攻击名为Target KPA（TKPA），利用图形理论分析在生成的图形中定位脆弱的节点，并使用LLM重写相应的叙述，实现对特定问答（QA）结果的精确控制，成功率为93.1%，同时保持有毒文本流畅自然。第二种攻击名为Universal KPA（UKPA），利用代词和依赖关系等语言线索通过改变具有全球影响力的单词来破坏生成图的结构完整性。修改的全文少于0.05%，QA准确性从95%下降到50%。此外，实验表明，最先进的防御方法无法检测到这些攻击，这凸显了保护GraphRAG管道免受知识中毒的保护在很大程度上仍然没有被探索。



